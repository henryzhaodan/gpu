import os
import torch
import collections
import cv2
import numpy as np
from math import cos, sin
from sklearn.decomposition import PCA


def plot_box(box, conf, img, show_txt=True):
    box = np.array(box, dtype=np.int32)
    txt = '{:.1f}'.format(conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 1)
    if show_txt:
        cv2.rectangle(img,
                      (box[0], box[1] - cat_size[1] - 2),
                      (box[0] + cat_size[0], box[1] - 2), color, -1)
        cv2.putText(img, txt, (box[0], box[1] - 2),
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_box(image, box, transcirpt, isFeaturemap=False):
    pts = box.astype(np.int)

    if isFeaturemap: # dimension reduction
        h, w, c = image.shape
        pca = PCA(n_components=3)
        ii = image.reshape(h*w, c)
        ii = pca.fit_transform(ii)

        for c in range(3):
            max = np.max(ii[:, c])
            min = np.min(ii[:, c])
            x_std = (ii[:, c] - min) / (max - min)
            ii[:, c] = x_std * 255
        image = ii.reshape(h, w, -1).astype(np.uint8)

    img = cv2.polylines(image, [pts], True, [150, 200, 200])

    origin = pts[0]
    font = cv2.FONT_HERSHEY_PLAIN
    img = cv2.putText(img, transcirpt, (origin[0], origin[1] - 10), font, 0.5, (255, 255, 255))

    cv2.imshow('text', img)
    cv2.waitKey()


class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(iter(self.alphabet)):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

        self.dict['-'] = len(self.dict)

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict.get(char.lower() if self._ignore_case else char, self.dict['-'])
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.tensor(text), torch.tensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length.item()
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def load_model(model, model_path, device):
    print("Loading checkpoint: {} ...".format(model_path))
    state_dict = torch.load(model_path, map_location='cpu')
    if not state_dict:
        raise RuntimeError('No checkpoint found.')
    # model.parallelize()
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def draw_marks(image, marks, color=(255, 255, 255)):
    """Draw mark points on image"""
    for mark in marks:
        cv2.circle(image, (int(mark[0]), int(mark[1])), 1, color, -1, cv2.LINE_AA)


def draw_gaze_angle(img, gaze_angle, gaze_tdx=None, gaze_tdy=None, gaze_size=10, color=(0, 0, 255)):
    if gaze_tdx is None and gaze_tdy is None:
        height, width, _ = img.shape
        gaze_tdx = width // 2
        gaze_tdy = height // 2
    gaze_yaw = gaze_angle[0]
    gaze_pitch = gaze_angle[1]
    gaze_yaw = gaze_yaw * np.pi / 180
    gaze_pitch = gaze_pitch * np.pi / 180
    # Z-Axis (out of the screen) drawn in blue
    x3 = gaze_size * (-np.cos(gaze_pitch) * np.sin(gaze_yaw)) + gaze_tdx
    y3 = gaze_size * (-np.sin(gaze_pitch)) + gaze_tdy
    cv2.arrowedLine(img, (int(gaze_tdx), int(gaze_tdy)), (int(x3), int(y3)), color, tipLength=0.1, thickness=1)

    return img
