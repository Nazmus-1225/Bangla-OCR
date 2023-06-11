from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

''' ["ka","kha","ga","gha","kna","cha","chha","ja","jha","yna","t`a","t`ha","d`a","d`ha","adna","ta","tha","da","dha","na","pa","pha","ba","bha","ma","yaw","ra","la","waw","sha","shat","sa","ha","aksha","tra","gya","0","1","2","3","4","5","6","7","8","9"]
labels =['yna', 't`aa', 't`haa', 'd`aa', 'd`haa', 'a`dna', 'ta', 'tha', 'da', 'dha', 'ka', 'na', 'pa', 'pha', 'ba', 'bha', 'ma', 'yaw', 'ra', 'la', 'waw', 'kha', 'sha', 'shat', 'sa', 'ha', 'aksha', 'tra', 'gya', 'ga', 'gha', 'kna', 'cha', 'chha', 'ja', 'jha', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
'''
#labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','ksha','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']
#
labels = [
    "া", "ি", "ী", "ু", "ূ", "ৃ", "ে", "ৈ", "ো", "ৌ", "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
    "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন", "প",
    "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ", "স", "হ", "ড়", "ঢ়", "য়", "ৎ", "ং", "ঃ", "ঁ", "ব্দ", "ঙ্গ",
    "স্ক", "স্ফ", "চ্ছ", "স্থ", "ক্ত", "স্ন", "ষ্ণ", "ম্প", "প্ত", "ম্ব", "ত্থ", "দ্ভ", "ষ্ঠ", "ল্প", "ষ্প",
    "ন্দ", "ন্ধ", "স্ম", "ণ্ঠ", "স্ত", "ষ্ট", "ন্ম", "ত্ত", "ঙ্খ", "ত্ন", "ন্ড", "জ্ঞ", "ড্ড", "ক্ষ", "দ্ব",
    "চ্চ", "ক্র", "দ্দ", "জ্জ", "ক্ক", "ন্ত", "ক্ট", "ঞ্চ", "ট্ট", "শ্চ", "ক্স", "জ্ব", "ঞ্জ", "দ্ধ", "ন্ন",
    "ঘ্ন", "ক্ল", "হ্ন", "০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "ল্ত", "স্প"
]
import numpy as np
from keras.preprocessing import image
test_image = cv2.imread("1686512374873.jpg")
image = cv2.resize(test_image, (32,32))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=3)
print("[INFO] loading network...")
import tensorflow as tf
model = tf.keras.models.load_model("BanglaOCR.h5")
lists = model.predict(image)[0]
print(lists)
for i, probability in enumerate(lists):
    if probability > 0.1:
        print("The letter is ",labels[i])

	