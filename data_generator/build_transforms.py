import data_generator.transforms as T
import data_generator.sltransforms as ST


def make_input_transform():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return normalize

def make_base_transforms(image_set):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600, keep_boxes = True),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            ST.RandomBlackAndWhite(prob = 0.1),
            ST.RandomSelectMulti([
                    ST.AdjustBrightness(0.7, 1.3),
                    ST.AdjustContrast(0.7, 1.3),
                    T.NoTransform(),
                ]),
        ])

    if image_set == "val":
        return T.NoTransform()

    raise ValueError(f"unknown {image_set}")

def make_tgt_transforms(image_set,
                        tgt_img_size=224,
                        tgt_img_max_size=448,
                        random_rotate = True,
                        use_sl_transforms = True,
                        random_perspective=True,
                        random_pad=True,
                        augmnet_bg="random"):
    if image_set == "train":
        tfs = []
        if random_rotate:
            tfs.append(
                T.RandomSelect(
                    T.RandomRotate(),
                    T.NoTransform(),
                ))
        if random_perspective:
            tfs.append(T.RandomPerspective())
        if random_pad:
            tfs.append(T.RandomPad((0.3, 0.3)))
        if type(augmnet_bg) == str:
            tfs.append(T.FillBackground(augmnet_bg))
        else:
            tfs.append(T.FillBackground("solid_color", augmnet_bg))
        tfs.append(T.Resize(tgt_img_size, max_size=tgt_img_max_size))
        tfs.append(T.RandomHorizontalFlip())
        if use_sl_transforms:
            tfs.append(
                ST.RandomSelectMulti([
                    ST.AdjustBrightness(0.7, 1.3),
                    ST.AdjustContrast(0.7, 1.3),
                    T.NoTransform(),
                ]),
            )
        tfs.append(
                ST.RandomSelectMulti([
                    ST.LightingNoise(),
                    T.NoTransform(),
                ]),
            )
        
        return T.Compose(tfs)
    
    if image_set == "val":
        tfs = []
        tfs.append(T.FillBackground("random", (124, 116, 104)))
        tfs.append(T.Resize(tgt_img_size, max_size=tgt_img_max_size))
        return T.Compose(tfs)

    raise ValueError(f"unknown {image_set}")