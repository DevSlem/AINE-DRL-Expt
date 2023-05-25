from argparse import ArgumentParser
from src import snake, bipedal_walker

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--env", type=str, help="environment: snake, ...")
    parser.add_argument("-i", "--inference", action="store_true", help="inference mode")
    args = parser.parse_args()
    env = args.env
    is_inference = args.inference
    
    if env == "snake":
        if not is_inference:
            snake.train()
        else:
            raise NotImplementedError
    elif env == "bipedal_walker":
        if not is_inference:
            bipedal_walker.train()
        else:
            bipedal_walker.inference()