import argparse
from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, required=True)
    parser.add_argument("checkpoint", type=str, required=True)
    parser.add_argument("input", type=str, required=True)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--score_thr", type=float, default=0.5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    model = init_detector(config_file, checkpoint_file, device="cuda:0")

    img = args.input
    result = inference_detector(model, img)

    model.show_result(
        img,
        result,
        score_thr=args.score_thr,
        out_file=args.out_file,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
    )


if __name__ == "__main__":
    main()
