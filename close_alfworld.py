import requests
import argparse


def main():
    env_url = args.url
    requests.post(env_url + "/close", json={})
    print("Closed alfworld")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:3000", type=str)
    args = parser.parse_args()
    main()
