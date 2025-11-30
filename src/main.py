from model_utils import load_model_once, live_inference_loop, MODEL_PATH

def main():
    model = load_model_once(MODEL_PATH)
    if model is None:
        exit()

    live_inference_loop(model)


if __name__ == "__main__":
    main()