from model_utils import load_model_once, live_inference_loop, MODEL_PATH, load_normalization_parameters
from data_utils import create_keypoint_param

def main():
    model = load_model_once(MODEL_PATH)
    if model is None:
        exit()

    load_normalization_parameters("data/keypoint_norm_params.npz")

    live_inference_loop(model)

def create_file():
    create_keypoint_param("data/X_keypoints.npy")

if __name__ == "__main__":
    main()