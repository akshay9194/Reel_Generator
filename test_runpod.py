"""
test_runpod.py — Smoke-test the RunPod serverless endpoint.

Usage:
    python test_runpod.py --face face.jpg --endpoint https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync

Set RUNPOD_API_KEY env var before running.
"""

import argparse
import base64
import json
import os
import sys
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face", required=True, help="Path to face image")
    parser.add_argument("--endpoint", required=True, help="RunPod endpoint URL")
    parser.add_argument("--prompt", default="beautiful indian girl in cafe, cinematic lighting, instagram style")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="result.mp4", help="Where to save the output video")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: Set RUNPOD_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    # Encode face image
    with open(args.face, "rb") as f:
        face_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "prompt": args.prompt,
            "face_image_base64": face_b64,
            "num_frames": args.frames,
        }
    }
    if args.seed is not None:
        payload["input"]["seed"] = args.seed

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Sending request to {args.endpoint} …")
    resp = requests.post(args.endpoint, json=payload, headers=headers, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data.get("output", {}):
        print("ERROR from RunPod:", data["output"]["error"], file=sys.stderr)
        sys.exit(1)

    video_b64 = data.get("output", {}).get("video_base64")
    if not video_b64:
        print("No video_base64 in response. Full response:")
        print(json.dumps(data, indent=2))
        sys.exit(1)

    with open(args.output, "wb") as f:
        f.write(base64.b64decode(video_b64))

    print(f"✅ Video saved → {args.output}")


if __name__ == "__main__":
    main()
