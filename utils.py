import random
import torch
import os
import time
import math
import numpy as np
import pprint as pprint
from sklearn import manifold
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()

def init_maps(args, class_order):
    book = args.cls_book
    st = args.base_class
    inc = args.way
    tot = args.num_classes

    tasks = []
    class_maps = []
    p = 0

    tasks.append(class_order[:st])
    class_map = np.full(tot, -1)
    for i, j in enumerate(tasks[-1]): class_map[j] = i
    class_maps.append(class_map)
    p += st

    while p < tot:
        tasks.append(class_order[p:p + inc])
        class_map = np.full(tot, -1)
        for i, j in enumerate(tasks[-1]): class_map[j] = i
        class_maps.append(class_map)
        p += inc
    book['tasks'] = [torch.tensor(task).cuda() for task in tasks]
    book['class_maps'] = [torch.tensor(class_map).cuda() for class_map in class_maps]
    #return tasks, class_maps

def inc_maps(args):
    book = args.cls_book
    tasks = book['tasks']
    num_classes = args.num_classes
    session = args.proc_book['session']

    prev = sorted(set([k for task in tasks[:session] for k in task]))
    prev_unsort = [k for task in tasks[:session] for k in task]
    seen = sorted(set([k for task in tasks[:session + 1] for k in task]))
    seen_unsort = [k for task in tasks[:session + 1] for k in task]
    prev_map = np.full(num_classes, -1)
    seen_map = np.full(num_classes, -1)
    prev_unsort_map = np.full(num_classes, -1)
    seen_unsort_map = np.full(num_classes, -1)
    for i, j in enumerate(prev): prev_map[j] = i
    for i, j in enumerate(seen): seen_map[j] = i
    for i, j in enumerate(prev_unsort): prev_unsort_map[j] = i
    for i, j in enumerate(seen_unsort): seen_unsort_map[j] = i

    book['prev'] = torch.tensor(prev, dtype=torch.long).cuda()
    book['prev_unsort'] = torch.tensor(prev_unsort, dtype=torch.long).cuda()
    book['seen'] = torch.tensor(seen, dtype=torch.long).cuda()
    book['seen_unsort'] = torch.tensor(seen_unsort, dtype=torch.long).cuda()
    book['prev_map'] = torch.tensor(prev_map).cuda()
    book['seen_map'] = torch.tensor(seen_map).cuda()
    book['prev_unsort_map'] = torch.tensor(prev_unsort_map).cuda()
    book['seen_unsort_map'] = torch.tensor(seen_unsort_map).cuda()
    #return prev_map, seen_map


def book_val(args):

    book_v = []
    num_classes = args.num_classes



    st = args.base_class
    inc = args.way
    tot = args.num_classes
    class_order = args.task_class_order

    tasks = []
    class_maps = []
    p = 0

    tasks.append(class_order[:st])
    class_map = np.full(tot, -1)
    for i, j in enumerate(tasks[-1]): class_map[j] = i
    class_maps.append(class_map)
    p += st

    while p < tot:
        tasks.append(class_order[p:p + inc])
        class_map = np.full(tot, -1)
        for i, j in enumerate(tasks[-1]): class_map[j] = i
        class_maps.append(class_map)
        p += inc
    tasks_ = [torch.tensor(task).cuda() for task in tasks]
    class_maps_ = [torch.tensor(class_map).cuda() for class_map in class_maps]

    #for session in range(args.start_session, args.sessions):
    for session in range(args.sessions):
        book_vs = {}
        book_vs['tasks'] = tasks_
        book_vs['class_maps'] = class_maps_

        tasks = book_vs['tasks']
        prev = sorted(set([k for task in tasks[:session] for k in task]))
        prev_unsort = [k for task in tasks[:session] for k in task]
        seen = sorted(set([k for task in tasks[:session + 1] for k in task]))
        seen_unsort = [k for task in tasks[:session + 1] for k in task]
        prev_map = np.full(num_classes, -1)
        seen_map = np.full(num_classes, -1)
        prev_unsort_map = np.full(num_classes, -1)
        seen_unsort_map = np.full(num_classes, -1)
        for i, j in enumerate(prev): prev_map[j] = i
        for i, j in enumerate(seen): seen_map[j] = i
        for i, j in enumerate(prev_unsort): prev_unsort_map[j] = i
        for i, j in enumerate(seen_unsort): seen_unsort_map[j] = i

        book_vs['prev'] = torch.tensor(prev, dtype=torch.long).cuda()
        book_vs['prev_unsort'] = torch.tensor(prev_unsort, dtype=torch.long).cuda()
        book_vs['seen'] = torch.tensor(seen, dtype=torch.long).cuda()
        book_vs['seen_unsort'] = torch.tensor(seen_unsort, dtype=torch.long).cuda()
        book_vs['prev_map'] = torch.tensor(prev_map).cuda()
        book_vs['seen_map'] = torch.tensor(seen_map).cuda()
        book_vs['prev_unsort_map'] = torch.tensor(prev_unsort_map).cuda()
        book_vs['seen_unsort_map'] = torch.tensor(seen_unsort_map).cuda()

        book_v.append(book_vs)

    return book_v

def learn_gauss(trainloader, model, base_class, num_features, seen_unsort_map,beta):
    # only model on gpu
    # Else given by cpu (torch)
    base_mean = torch.zeros(base_class, num_features)
    base_cov = torch.zeros(base_class, num_features, num_features)
    embedding_list = []
    label_list = []
    seen_unsort_map_ = seen_unsort_map.cpu()
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)
            embedding = torch.pow(embedding,beta)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    for i in range(base_class):
        ind_cl = torch.where(i == seen_unsort_map_[label_list])[0]
        base_mean[i] = embedding_list[ind_cl].mean(dim=0)
        mat = embedding_list[ind_cl] - embedding_list[ind_cl].mean(dim=0)  # 500,512
        mat = mat.unsqueeze(dim=2)  # 500,512,1
        mat2 = mat.permute(0, 2, 1)  # 500,1,512
        cov_ = torch.bmm(mat, mat2)  # 500,512,512
        cov_ = torch.sum(cov_,dim=0)/(len(cov_)-1)
        base_cov[i] = cov_
    return base_mean, base_cov

def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    # torch cpu
    dist = []
    for i in range(len(base_means)):
        dist.append(torch.norm(query - base_means[i]))
    index = torch.topk(torch.tensor(dist),k).indices
    slc_base_means = torch.index_select(base_means,dim=0,index=index)
    mean = torch.cat([slc_base_means, query.unsqueeze(0)])
    calibrated_mean = torch.mean(mean, dim=0)
    slc_base_covs = torch.index_select(base_cov,dim=0,index=index)
    calibrated_cov = torch.mean(slc_base_covs, dim=0) + alpha

    return calibrated_mean, calibrated_cov

def distribution_calibration2(query, base_means, base_cov, k, alpha=0.21):
    # torch cpu
    dist = []
    for i in range(len(base_means)):
        dist.append(torch.norm(query - base_means[i]))
    index = torch.topk(torch.tensor(dist),k).indices.cuda()
    slc_base_covs =torch.index_select(base_cov,dim=0,index=index)
    calibrated_cov = torch.mean(slc_base_covs, dim=0) + alpha
    return calibrated_cov


#def checkparser_dependencies(args):

def f(d,n):
    x = math.pow(n,-(2/(d-1)))
    y = math.gamma(1+1/(d-1))
    z = (math.gamma(d/2)/(2*math.sqrt(math.pi)*(d-1)*math.gamma((d-1)/2)))
    return x*y*(math.pow(z,-(1/(d-1))))


def tot_datalist(dataloader, model, map=None, gpu=False):
    # model, map is assumed to be in gpu

    data_ = []
    label_ = []
    with torch.no_grad():
        model.eval()
        model.module.set_mode('encoder')
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = model(data).detach()
            if gpu==True:
                data_.append(data)
                label_.append(label)
            else:
                data_.append(data.cpu())
                label_.append(label.cpu())
        data_ = torch.cat(data_, dim=0)
        label_ = torch.cat(label_, dim=0)
        if map is not None:
            if gpu==True:
                label_cls = (map)[label_]
            else:
                label_cls = (map.cpu())[label_]
        else:
            label_cls = label_
        #data_ = np.array(data_)
        #label_cls = np.array(label_cls)
        return data_, label_cls

def draw_tsne(data_, label_, n_components, perplexity,palette,num_class, title=None):

    tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
    x = tsne.fit_transform(data_)
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    #c=palette[label_.astype(np.int)])
                    c=palette[torch.tensor(label_,dtype=int)])
    plt.title(title)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(num_class):
        # Position of each label.
        xtext, ytext = np.median(x[label_ == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()

def set_trainable_module(moduelist_, ts=[]):
    if not isinstance(ts, (list, range)):
        ts = [ts]
    for t, m in enumerate(moduelist_):
        requires_grad = (t in ts)
        for param in m.parameters():
            param.requires_grad = requires_grad

def set_trainable_param(paramlist_, ts=[]):
    if not isinstance(ts, (list, range)):
        ts = [ts]
    for t, m in enumerate(paramlist_):
        requires_grad = (t in ts)
        m.requires_grad = requires_grad

def save_obj(trlog, gauss_book):
    dict = {}
    dict['trlog']=trlog
    dict['gauss_book']=gauss_book
    with open('obj.pkl','wb') as f:
        pickle.dump(dict,f,pickle.HIGHEST_PROTOCOL)

def save_obj(trlog, gauss_book,fn):
    dict = {}
    dict['trlog']=trlog
    dict['gauss_book']=gauss_book
    with open(fn,'wb') as f:
        pickle.dump(dict,f,pickle.HIGHEST_PROTOCOL)
    print('save object saved')

def repr_tot(session, angle_w):
    angle_w_tot = []
    for i in range(session+1):
        angle_w_tot.append(angle_w[i])
    angle_w_tot = torch.cat(angle_w_tot,dim=0)
    return angle_w_tot

def cosine_distance(input1, input2):
    if len(input1.shape)>1 and len(input2.shape)>1:
        return F.linear(F.normalize(input1), F.normalize(input2))
    else:
        return F.linear(input1/torch.norm(input1), input2/torch.norm(input2))


def cos2angle(cosine):
    return torch.acos(cosine.clamp(-1,1)) * 180/math.pi