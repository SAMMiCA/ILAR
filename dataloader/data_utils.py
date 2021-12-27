import numpy as np
import torch
from dataloader.sampler import CategoriesSampler

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.num_classes = 100
        args.width = 32
        args.height = 32

        if args.base_class == None and args.way == None:
            args.base_class = 60
            args.way = 5
            # args.shot = 5
            args.sessions = 9
        elif not args.base_class == None and not args.way == None:
            args.sessions = 1 + int((args.num_classes-args.base_class)/args.way)
        else:
            raise NotImplementedError

    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.num_classes = 200
        args.width = 224
        args.height = 224

        if args.base_class == None and args.way == None:
            args.base_class = 100
            args.way = 10
            # args.shot = 5
            args.sessions = 11
        elif not args.base_class == None and not args.way == None:
            args.sessions = 1 + int((args.num_classes-args.base_class)/args.way)
        else:
            raise NotImplementedError

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.num_classes = 100
        args.width = 84
        args.height = 84

        if args.base_class == None and args.way == None:
            args.base_class = 60
            args.way = 5
            # args.shot = 5
            args.sessions = 9
        elif not args.base_class == None and not args.way == None:
            args.sessions = 1 + int((args.num_classes-args.base_class)/args.way)
        else:
            raise NotImplementedError

    args.Dataset=Dataset
    return args


def get_dataloader(args, session=None):
    if session==None:
        session = args.proc_book['session']
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
        return trainset, trainloader, testloader
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
        return trainset, trainloader, testloader


def get_base_dataloader(args):
    class_index = np.array(args.cls_book['tasks'][0].cpu())
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(train=True, base_sess=True,
                                       root=args.dataroot, download=True, index=class_index)
        testset = args.Dataset.CIFAR100(train=False, base_sess=True,
                                      root=args.dataroot, download=False, index=class_index)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(train=True, base_sess=True,
                                       root=args.dataroot, index=class_index)
        testset = args.Dataset.CUB200(train=False, base_sess=True,
                                      root=args.dataroot, index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(train=True, base_sess=True,
                                       root=args.dataroot, index=class_index)
        testset = args.Dataset.MiniImageNet(train=False, base_sess=True,
                                      root=args.dataroot, index=class_index)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session, coreset=None, core_labels=None):
    #txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    class_new_tr, class_new_te = get_session_classes(args, session)
    if args.dataset == 'cifar100':
        #trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
        #                                 index=class_index, base_sess=False)
        trainset = args.Dataset.CIFAR100(train=True, base_sess=False,
                                       root=args.dataroot, download=False, index=class_new_tr, shot = args.shot)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(train=True, base_sess=False,
                                       root=args.dataroot, index=class_new_tr, shot = args.shot)

    if args.dataset == 'mini_imagenet':
        #trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
        #                                     index_path=txt_path)
        trainset = args.Dataset.MiniImageNet(train=True, base_sess=False,
                                       root=args.dataroot, index=class_new_tr, shot = args.shot)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new,
                                                  shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)


    if args.dataset == 'cifar100':
        #testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
        #                                index=class_new, base_sess=False)
        testset = args.Dataset.CIFAR100(train=False, base_sess=False,
                                      root=args.dataroot, download=False, index=class_new_te)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(train=False, base_sess=False,
                                      root=args.dataroot, index=class_new_te)
    if args.dataset == 'mini_imagenet':
        #testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
        #                                    index=class_new)
        testset = args.Dataset.MiniImageNet(train=False, base_sess=False,
                                      root=args.dataroot, index=class_new_te)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader


def get_session_classes(args,session):
    class_list_tr =np.array(args.cls_book['tasks'][session].cpu())
    class_list_te = np.array(args.book_v[session]['seen_unsort'].cpu())
    return class_list_tr, class_list_te






