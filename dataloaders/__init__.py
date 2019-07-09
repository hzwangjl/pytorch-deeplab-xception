from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, mydataset
from torch.utils.data import DataLoader
from torchvision import transforms

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'mydataset':
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.519401, 0.359217, 0.310136], [0.061113, 0.048637, 0.041166]),#R_var is 0.061113, G_var is 0.048637, B_var is 0.041166
        ])

        valid_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.517446, 0.360147, 0.310427], [0.061526,0.049087, 0.041330])#R_var is 0.061526, G_var is 0.049087, B_var is 0.041330
        ])

        train_set = mydataset.SegmentDataset(args, split='train', transform=train_transform)
        valid_set = mydataset.SegmentDataset(args, split='valid', transform=valid_transform)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, valid_loader, test_loader, num_class

    else:
        raise NotImplementedError

