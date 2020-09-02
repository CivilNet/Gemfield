from torchvision import transforms as trans
from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        #print("gemfieldimg: ", tuple_with_path[0],'\t',tuple_with_path[1],'\t',tuple_with_path[2])
        return tuple_with_path

def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolderWithPaths(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num


ds, class_num = get_train_dataset('gemfield_imgs_dir')
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=3)

for i, (imgs, labels) in enumerate(loader):
    print("do your thing...")

