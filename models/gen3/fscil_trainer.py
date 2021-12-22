from models.base.base import Trainer
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import math
import torch.nn as nn

from tqdm import tqdm

import copy
from copy import deepcopy
from sklearn import manifold

from utils import *
from dataloader.data_utils import *
from tensorboardX import SummaryWriter
from .Network import MYNET


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        if self.args.use_selfsup_fe:
            self.model.load_selfsup_weight(self.args)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        self.writer = SummaryWriter(os.path.join(self.args.log_path, self.args.project))

        self.args.task_class_order = np.random.permutation(np.arange(self.args.num_classes)).tolist()
        self.args.cls_book = {}
        self.args.book_v = book_val(self.args)
        self.args.proc_book = {}
        self.args.coreset = None
        self.args.eps = 1e-7

        self.args.gauss_book={}

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())


    def get_optimizer_base(self):
        if self.args.freeze_backbone:
            for param in self.model.module.encoder.parameters():
                param.requires_grad = False

        if not self.args.angle_mode==None:
            for param in self.model.module.fc.parameters():
                param.requires_grad = False
            set_trainable_param(self.model.module.angle_w, [0])

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.module.parameters()),
                                    self.args.lr_base, momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        return optimizer, scheduler

    def get_optimizer_new(self):
        assert self.args.angle_mode is not None

        for param in self.model.module.encoder.parameters():
            param.requires_grad = True
        for param in self.model.module.fc.parameters():
            param.requires_grad = False
        set_trainable_param(self.model.module.angle_w, [self.args.proc_book['session']])

        enc_param_list = list(kv[0] for kv in self.model.module.encoder.named_parameters())
        enc_param_list = ['encoder.' + k for k in enc_param_list]

        enc_params = list(filter(lambda kv: kv[0] in enc_param_list and kv[1].requires_grad,
                                 self.model.module.named_parameters()))
        else_params = list(filter(lambda kv: kv[0] not in enc_param_list and kv[1].requires_grad,
                                  self.model.module.named_parameters()))

        enc_params = [i[1] for i in enc_params]
        else_params = [i[1] for i in else_params]

        optimizer = torch.optim.SGD([{'params': enc_params, 'lr': self.args.lr_new_enc},
                                     {'params': else_params, 'lr': self.args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)

        if self.args.schedule_new == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_new, gamma=self.args.gamma)
        elif self.args.schedule_new == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones_new,
                                                             gamma=self.args.gamma)
        return optimizer, scheduler

    def get_coreset(self, trainloader, label_list, sess=None):
        if sess is None:
            sess = self.args.proc_book['session']
        candi_ = torch.zeros(len(self.args.cls_book['tasks'][sess]), 3, self.args.width, self.args.height)
        candi_ = candi_.cuda()
        angle_w_tot = repr_tot(sess, self.model.module.angle_w)

        self.model.eval()
        self.model.module.set_mode('encoder')
        # standard classification for pretrain
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, train_label = [_.cuda() for _ in batch]
                feats = self.model(data)
                for idx, j in enumerate(label_list):
                    ind_cl = torch.where(j==train_label)[0]
                    repr_order = self.args.cls_book['seen_unsort_map'][j]
                    for k in ind_cl:
                        if sum(angle_w_tot[repr_order] == torch.zeros(angle_w_tot[repr_order].shape).cuda()) == 0:
                            candi_[idx] = data[k]
                        else:
                            if cosine_distance(feats[k],angle_w_tot[repr_order]) < \
                                    cosine_distance(self.model(candi_[idx].unsqueeze(dim=0)),angle_w_tot[repr_order]):
                                candi_[idx] = data[k]

        return candi_

    def ang_triplet_center_loss(self, feat, label, repr, m_atcl):
        # x.shape: B x n_feat
        # y.shape: B, with 0~n_class-1 value
        # repr.shape: n_class x n_feat
        # Note that label values are assumed to have values from 0 to n_class-1, == cls version.
        B = feat.shape[0]
        n_feat = feat.shape[1]
        n_class = repr.shape[0]
        feat_norm = feat / feat.norm(dim=1).unsqueeze(dim=1).expand(feat.shape)
        repr_norm = repr / repr.norm(dim=1).unsqueeze(dim=1).expand(repr.shape)

        cos_dist = torch.mm(feat_norm, repr_norm.t()).clamp(-1+self.args.eps, 1-self.args.eps) # B x n_class
        theta = torch.acos(cos_dist)
        loss = 0.0
        for i in range(B):
            sum = 0.0
            sum += theta[i, label[i]]
            min_idx_order = torch.topk(theta[i], k=2, largest=False)
            min = int(min_idx_order[0][0])
            if min != label[i]:
                sum -= theta[i, min]
            else:
                sum -= theta[i, int(min_idx_order[0][1])]

            sum += m_atcl
            loss += max(sum,0.0)
        return loss

    def loss_dist(self, feature1, feature2):
        assert len(feature1) == len(feature2)
        n_class = len(feature1)
        sum=0
        for i in range(n_class):
            norm1 = feature1[i] / torch.norm(feature1[i])
            norm2 = feature2[i] / torch.norm(feature2[i])
            sum += torch.pow(torch.norm(norm1-norm2),2)
        return sum/n_class

    def update_repr(self, prev_model):
        session = self.args.proc_book['session']

        with torch.no_grad():
            idx = 0
            for sess in range(session):

                #for i in range(len(self.args.repr[sess])):
                for i in range(len(self.model.module.angle_w[sess])):
                    prev_model.module.set_mode('encoder')
                    self.model.module.set_mode('encoder')
                    prev_feat = prev_model(self.args.coreset[idx + i].unsqueeze(dim=0))
                    cur_feat = self.model(self.args.coreset[idx + i].unsqueeze(dim=0))
                    self.model.module.angle_w[sess][i] += (cur_feat-prev_feat).squeeze()
                idx += len(self.model.module.angle_w[sess])

    def base_train(self, trainloader, optimizer, scheduler, epoch):
        tl = Averager()
        ta = Averager()
        self.model.train()
        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):
            self.args.proc_book['step'] += 1
            data, train_label = [_.cuda() for _ in batch]

            self.model.module.set_mode('encoder')
            feat = self.model(data)

            self.model.module.set_mode('get_logit_cos')
            target_cls = self.args.cls_book['class_maps'][0][train_label]
            logits_cls, cosine_cls = self.model(feat, target_cls, self.args.proc_book['session'])

            loss = F.cross_entropy(logits_cls, target_cls)
            acc = count_acc(cosine_cls, target_cls)
            total_loss = loss

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},loss={:.4f},acc={:.4f}'.format(epoch, lrc, loss, acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def new_train(self, train_set, trainloader, optimizer, scheduler, result_list, prev_model):
        session = self.args.proc_book['session']
        print("training session: [%d]" % session)
        assert self.args.angle_mode is not None
        assert self.args.varsess == True
        assert self.args.not_data_init == False

        tl = Averager()
        ta = Averager()
        self.model.module.mode = self.args.angle_mode
        self.model.eval()

        session = self.args.proc_book['session']

        for epoch in range(self.args.epochs_new):
            self.args.proc_book['epoch'] += 1
            self.args.proc_book['step'] += 1

            self.model.module.set_mode('encoder')
            if self.args.batch_size_new == 0:
                for batch in trainloader:
                    data, label = [_.cuda() for _ in batch]
                    cur_feat=self.model(data)
                    cur_label = label
            else:
                cur_feat = []
                cur_label = []
                for batch in trainloader:
                    data, label = [_.cuda() for _ in batch]
                    data = self.model(data)
                    cur_feat.append(data)
                    cur_label.append(label)
                cur_feat = torch.cat(cur_feat, dim=0)
                cur_label = torch.cat(cur_label, dim=0)
            cur_label_cls = self.args.cls_book['seen_unsort_map'][torch.tensor(cur_label,dtype=int)]

            prev_model.module.set_mode('encoder')
            if self.args.batch_size_new == 0:
                for batch in trainloader:
                    data, label = [_.cuda() for _ in batch]
                    prev_feat=prev_model(data)
                    prev_label = label
            else:
                prev_feat = []
                prev_label = []
                for batch in trainloader:
                    data, label = [_.cuda() for _ in batch]
                    data = prev_model(data)
                    prev_feat.append(data)
                    prev_label.append(label)
                prev_feat = torch.cat(prev_feat, dim=0)
                prev_label = torch.cat(prev_label, dim=0)

            prev_label_cls = self.args.cls_book['seen_unsort_map'][torch.tensor(prev_label, dtype=int)]

            if  epoch == 0:
                with torch.no_grad():
                    for i in range(self.args.way):
                        idx = self.args.cls_book['tasks'][session][i]
                        data_index = (cur_label == idx).nonzero().squeeze(-1)
                        embedding = cur_feat[data_index]
                        proto = embedding.mean(0)
                        self.model.module.angle_w[session][i] = proto.detach()

            beta = self.args.tukey_beta

            temp_repr = repr_tot(session, self.model.module.angle_w),
            for i in range(len(temp_repr)):
                temp_repr[i]/ torch.norm(temp_repr[i])

            with torch.no_grad():
                # torch cpu
                base_means = self.args.gauss_book['base_mean']
                base_cov = self.args.gauss_book['base_cov']
                num_sampled = self.args.num_sampled

                data_ = cur_feat.cpu()
                label_ = cur_label.cpu()
                sampled_data = []
                sampled_label = []
                for i in range(len(data_)):
                    mean, cov = distribution_calibration(torch.pow(data_[i],beta), base_means, base_cov, k=2)
                    sampled = torch.tensor(np.random.multivariate_normal(mean=mean,cov=cov, size=num_sampled)).float()
                    sampled = torch.pow(sampled,1/beta)
                    sampled_data.append(sampled)
                    sampled_label.extend([label_[i]] * num_sampled)
                sampled_data = torch.cat(sampled_data,dim=0)
                sampled_label = torch.tensor(sampled_label)

            cur_feat_aug = torch.cat([cur_feat, sampled_data.cuda().detach()], dim=0)
            cur_label_aug = torch.cat([cur_label, sampled_label.cuda().detach()], dim=0)
            cur_label_aug_cls = self.args.cls_book['seen_unsort_map'][torch.tensor(cur_label_aug, dtype=int)]

            self.model.module.set_mode('encoder')
            cur_core_feat = self.model(self.args.coreset)
            cur_core_label = torch.arange(cur_core_feat.shape[0]).cuda()

            prev_model.module.set_mode('encoder')
            with torch.no_grad():
                prev_core_feat = prev_model(self.args.coreset)
                prev_core_label = torch.arange(prev_core_feat.shape[0]).cuda()

            base_means = self.args.gauss_book['base_mean'].cuda().detach()
            base_cov = self.args.gauss_book['base_cov'].cuda().detach()
            num_sampled_n = self.args.num_sampled_n

            sampled_data = []
            sampled_label = []
            for i in range(len(cur_core_feat)):
                cov = distribution_calibration2(cur_core_feat[i], base_means, base_cov, k=2)
                mvn = torch.distributions.MultivariateNormal(cur_core_feat[i],cov)
                sampled = mvn.sample((num_sampled_n,))
                sampled = torch.pow(sampled,1/beta)
                sampled_data.append(sampled)
                sampled_label.extend([cur_core_label[i]] * num_sampled_n)

            if num_sampled_n>0:
                sampled_data = torch.cat(sampled_data,dim=0)
                sampled_label = torch.tensor(sampled_label).cuda()
                cur_core_feat_aug = torch.cat([cur_core_feat, sampled_data],dim=0)
                cur_core_label_aug = torch.cat([cur_core_label, sampled_label],dim=0)
            else:
                cur_core_feat_aug = cur_core_feat
                cur_core_label_aug = cur_core_label

            self.model.module.set_mode('get_logit_cos')
            logits_cls, cosine_cls = self.model(cur_feat_aug, cur_label_aug_cls, session)
            logits_cls_core_befsess, cosine_cls_core_befsess = self.model(cur_core_feat, cur_core_label, session-1)
            _, cosine_cls_core_aug_befsess = self.model(cur_core_feat_aug, cur_core_label_aug, session-1)

            prev_model.module.set_mode('get_logit_cos')
            _, prev_cosine_cls_core_befsess = prev_model(prev_core_feat, prev_core_label,
                                                          session - 1)

            #loss_distill = F.cross_entropy(cosine_cls[:len(cur_feat)], prev_cosine_cls.long())

            loss = F.cross_entropy(cosine_cls, cur_label_aug_cls)
            loss_prev_ce = F.cross_entropy(cosine_cls_core_befsess, cur_core_label)

            loss_ce5 = F.cross_entropy(cosine_cls_core_aug_befsess, cur_core_label_aug)

            loss_dist = self.loss_dist(cur_core_feat, prev_core_feat)
            loss_dist2 = self.loss_dist(prev_feat, cur_feat)

            loss_distill4 = F.kl_div(cosine_cls_core_befsess.log_softmax(dim=1),
                                    prev_cosine_cls_core_befsess.softmax(dim=1))

            loss_atcl = self.ang_triplet_center_loss(cur_feat_aug, cur_label_aug_cls,
                                                     repr_tot(session, self.model.module.angle_w),
                                                     self.args.m_atcl_n)

            acc = count_acc(cosine_cls[:len(cur_feat)], cur_label_aug_cls[:len(cur_feat)])

            loss =        self.args.w_ce*loss \
                        + self.args.w_prev_ce*loss_prev_ce \
                        + self.args.w_ce5*loss_ce5 \
                        + self.args.w_l2*loss_dist \
                        + self.args.w_l22*loss_dist2 \
                        + self.args.w_distill4 * loss_distill4 \
                        + self.args.w_atcl_n*loss_atcl

            total_loss = loss
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass

        print('Incremental session, test')

        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def test(self, testloader):
        epoch = self.args.proc_book['epoch']
        session = self.args.proc_book['session']

        if not self.args.varsess:
            test_class = self.args.base_class + session * self.args.way
        self.model.eval()
        vl = Averager()
        va = Averager()
        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]

                assert self.args.varsess == True
                self.model.module.set_mode(self.args.angle_mode)

                target_cls = self.args.cls_book['seen_unsort_map'][test_label]
                logits_cls, cosine_cls = self.model(data, target_cls, session)
                loss = F.cross_entropy(logits_cls, target_cls)
                acc = count_acc(cosine_cls, target_cls)
                total_loss = loss

                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        return vl, va

    def train(self):
        assert self.args.angle_mode is not None
        assert self.args.varsess == True
        assert self.args.not_data_init == False

        sns.set_style('darkgrid')
        sns.set_palette('muted')
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        palette = np.array(sns.color_palette("hls", self.args.num_classes))

        n_components = 2
        perplexity = 30
        t_start_time = time.time()

        if self.args.varsess:
            init_maps(self.args, self.args.task_class_order)

        if self.args.model_dir is not None:
            self.args.proc_book['session'] = self.args.start_session-1
            for i in range(self.args.start_session):
                inc_maps(self.args)
            with open(self.args.obj_root, 'rb') as f:
                dict_ = pickle.load(f)
                self.trlog = dict_['trlog']
                self.args.gauss_book = dict_['gauss_book']
        else:
            self.args.proc_book['session'] = -1


        # init train statistics
        result_list = [self.args]
        # args.seesions: total session num. == len(self.tasks)
        natsa_ = []
        for session in range(self.args.start_session, self.args.sessions):
            self.args.proc_book['step'] = -1
            self.args.proc_book['epoch'] = -1

            self.args.proc_book['session'] += 1
            if self.args.varsess:
                inc_maps(self.args)

            if session == 0:
                train_set, trainloader, testloader = get_dataloader(self.args)
            else:
                train_set, trainloader, testloader = get_dataloader(self.args)

            self.model.load_state_dict(self.best_model_dict)


            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(self.args.epochs_base):
                    self.args.proc_book['epoch'] += 1
                    start_time = time.time()
                    # train base sess
                    tl, ta = self.base_train(trainloader, optimizer, scheduler, epoch)
                    # test model with all seen class
                    tsl, tsa = self.test(testloader)

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (self.args.epochs_base - epoch) / 60))
                    scheduler.step()

                if (tsa * 100) >= self.trlog['max_acc'][session]:
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    self.trlog['max_acc_epoch'] = epoch
                    save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)
                    torch.save(optimizer.state_dict(), os.path.join(self.args.save_path, 'optimizer_best.pth'))
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    print('********A better model is found!!**********')
                    print('Saving model to :%s' % save_model_dir)
                print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                   self.trlog['max_acc'][session]))

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))


                if not self.args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = self.replace_base_fc(train_set, testloader.dataset.transform,
                                                      self.args.angle_mode is not None)
                    best_model_dir_root = os.path.join(self.args.save_path, 'rep_base_fc')
                    ensure_path(best_model_dir_root)
                    best_model_dir = os.path.join(best_model_dir_root,'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    print('After all epochs, test')
                    tsl, tsa = self.test(testloader)
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                self.args.gauss_book['base_mean'], self.args.gauss_book['base_cov'] = \
                    learn_gauss(trainloader, self.model, self.args.base_class, self.model.module.num_features,
                                     self.args.cls_book['seen_unsort_map'].cpu(),self.args.tukey_beta)

            else:  # incremental learning sessions
                assert self.args.angle_mode is not None
                assert self.args.varsess == True
                print("training session: [%d]" % session)

                if self.args.coreset is None:
                    # This is maybe we loaded from pre-trained model.
                    # Assume We start from session 1. (not 0)
                    if self.args.core_root is None:
                        _, base_trainloader, _ = get_dataloader(self.args, session=0)
                        self.args.coreset = self.get_coreset(base_trainloader, self.args.cls_book['tasks'][0],
                                                         sess=0).cuda()
                        dict_ = {'coreset': self.args.coreset}
                        with open('coreset.pkl', 'wb') as f:
                            pickle.dump(dict_, f, pickle.HIGHEST_PROTOCOL)
                        print('save: coreset saved')
                    else:
                        with open(self.args.core_root, 'rb') as f:
                            dict_ = pickle.load(f)
                            self.args.coreset = dict_['coreset']

                optimizer, scheduler = self.get_optimizer_new()
                prev_model = copy.deepcopy(self.model)
                for param in prev_model.module.parameters():
                    param.requires_grad = False
                self.model.module.set_mode(self.args.new_mode)
                self.model.eval()
                prev_model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                tl, ta = self.new_train(train_set,trainloader,optimizer, scheduler, result_list, prev_model)
                print('Incremental session, test')
                self.update_repr(prev_model)
                tsl, tsa = self.test(testloader)

                candi_ = self.get_coreset(trainloader, self.args.cls_book['tasks'][session])
                self.args.coreset = torch.cat((self.args.coreset,candi_),dim=0)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

    def replace_base_fc(self, trainset, transform, angle_mode=False):
        # replace fc.weight with the embedding average of train data
        assert angle_mode == True
        self.model.eval()

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                  num_workers=8, pin_memory=True, shuffle=False)
        trainloader.dataset.transform = transform
        embedding_list, label_list = tot_datalist(trainloader, self.model, map=None, gpu=False)
        label_list = label_list.cuda()

        proto_list = []

        for class_index in self.args.cls_book['tasks'][0]:
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)

        self.model.module.angle_w[0].data = proto_list.cuda()

        return self.model