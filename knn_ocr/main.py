import cv2
import numpy as np
import os
from pathlib import Path

test_path = Path("./task")
train_path = test_path / "train"


def extract_features(img_crop):
    h, w = img_crop.shape
    if h > w:
        m = h
    else:
        m = w
    square = np.zeros((m, m), dtype=np.uint8)

    dy = (m - h) // 2
    dx = (m - w) // 2
    square[dy:dy + h, dx:dx + w] = img_crop

    resized = cv2.resize(square, (20, 20))
    return np.array(resized.flatten(), dtype="f4")

def make_train(path):
    train = []
    responses = []
    chararr = []

    folders = sorted(os.listdir(path))
    class_id = 0

    for folder_name in folders:
        folder_path = path / folder_name
        if folder_path.is_dir():
            chararr.append(folder_name[-1])

            for p in sorted(folder_path.glob("*.png")):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
                    x, y, w, h = cv2.boundingRect(cv2.findNonZero(binary))

                    train.append(extract_features(binary[y:y + h, x:x + w]))
                    responses.append(class_id)
            class_id += 1

    train_matrix = np.array(train, dtype="f4")
    responses_matrix = np.array(responses, dtype="f4").reshape(-1, 1)
    return train_matrix, responses_matrix, chararr


def get_merged_bboxes(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 2 and h > 2:
            rects.append([x, y, w, h])

    rects.sort(key=lambda item: item[0])
    if len(rects) == 0:
        return []

    merged = [rects[0]]

    for i in range(1, len(rects)):
        last = merged[-1]
        curr = rects[i]

        last_x, last_y, last_w, last_h = last
        curr_x, curr_y, curr_w, curr_h = curr

        if curr_x < (last_x + last_w / 2):
            min_x = min(last_x, curr_x)
            min_y = min(last_y, curr_y)
            max_x = max(last_x + last_w, curr_x + curr_w)
            max_y = max(last_y + last_h, curr_y + curr_h)

            merged[-1] = [min_x, min_y, max_x - min_x, max_y - min_y]
        else:
            merged.append(curr)

    return merged


def main():
    train, responses, chararr = make_train(train_path)

    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, responses)

    for i in range(7):
        img_path = test_path / f"{i}.png"
        if not img_path.exists():
            continue

        gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        rects = get_merged_bboxes(binary)

        sum_width = 0
        for r in rects:
            sum_width += r[2]
        avg_width = sum_width / len(rects) if len(rects) > 0 else 0

        recognized_text = ""

        for idx in range(len(rects)):
            x, y, w, h = rects[idx]

            if idx > 0:
                prev_x, _, prev_w, _ = rects[idx - 1]
                gap = x - (prev_x + prev_w)
                if gap > avg_width * 0.4:
                    recognized_text += " "

            features = extract_features(binary[y:y + h, x:x + w]).reshape(1, -1)
            _, result, _, _ = knn.findNearest(features, 3)

            answer_idx = int(result[0][0])
            recognized_text += chararr[answer_idx]

        print(f"Изображение {i}: {recognized_text}")


if __name__ == "__main__":
    main()