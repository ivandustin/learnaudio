import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import random as rnd
import numpy
import neural
from keras.models import load_model

MIN         = -32768
MAX         = 32767
INPUT_SIZE  = 100
NOISE_RANGE = 1000
MODEL_PATH  = "model.h5"

def get_L(samples):
    return samples[0::2]

def get_R(samples):
    return samples[1::2]

def sample(sound):
    return get_L(sound.get_array_of_samples())[1000:1004]

def bands(sound):
    samples = sound.get_array_of_samples()
    return (samples[0::2], samples[1::2])

def plot_sample(sound):
    (L, R) = bands(sound)
    # plt.plot(L[0:1000])
    plt.plot(L)
    plt.show()

def noise(a):
    for i in range(0, len(a)):
        v = a[i]
        r = rnd.randint(-NOISE_RANGE, NOISE_RANGE)
        n = v + r
        n = min(n, MAX)
        n = max(n, MIN)
        if (v > 0):
            a[i] = n
    return a

def normalize(a):
    o = []
    for i in range(0, len(a)):
        v = a[i]
        o.append((v + 32768) / (32768*2))
    return o

def denormalize(a):
    o = []
    for i in range(0, len(a)):
        v = a[i]
        n = int((v * (32768*2)) - 32768)
        o.append(n)
    return o

def prepare_x_y(sound):
    (L, R) = bands(sound)
    n = int(len(L) / INPUT_SIZE)
    x = []
    y = []
    for i in range(0, n):
        a = normalize(noise(L[i*INPUT_SIZE:(i*INPUT_SIZE)+INPUT_SIZE]))
        b = normalize(L[i*INPUT_SIZE:(i*INPUT_SIZE)+INPUT_SIZE])
        x.append(numpy.array(a + [1]))
        y.append(numpy.array(b))
    return (numpy.array(x), numpy.array(y))

def prepare_y(sound):
    (L, R) = bands(sound)
    n = int(len(L) / INPUT_SIZE)
    l = []
    r = []
    for i in range(0, n):
        a = normalize(L[i*INPUT_SIZE:(i*INPUT_SIZE)+INPUT_SIZE])
        b = normalize(R[i*INPUT_SIZE:(i*INPUT_SIZE)+INPUT_SIZE])
        l.append(numpy.array(a + [1]))
        r.append(numpy.array(b + [1]))
    return (numpy.array(l), numpy.array(r))

def unpack(x):
    return [a.tolist() for a in x.tolist()]

def merge(l, r):
    o = []
    for i in range(0, len(l)):
        o.append(l[i])
        o.append(r[i])
    return o

def flatten(l):
    return [ v for a in l for v in a ]

def overlay(a, b):
    for i in range(0, len(b)):
        if b[i] > MAX:
            a[i] = MAX
        elif b[i] < MIN:
            a[i] = MIN
        else:
            a[i] = b[i]
    return a

def info(mp3):
    sound = AudioSegment.from_mp3(mp3)
    print("channels", sound.channels)
    print("frame rate", sound.frame_rate)
    print("sample width", sound.sample_width)
    print("max amplitude", sound.max)
    print("duration(minutes)", len(sound) / 1000 / 60)
    print("dBFS", sound.dBFS)
    samples = sound.get_array_of_samples()
    print("sample length", len(samples))
    print("sample", sample(sound))

def prepare_model():
    if (not os.path.exists(MODEL_PATH)):
        model = neural.network(101, [300, 250, 200, 150], 100)
        model.save(MODEL_PATH)

def train(mp3):
    sound = AudioSegment.from_mp3(mp3)
    (x, y) = prepare_x_y(sound)
    model = load_model(MODEL_PATH)
    try:
        model.fit(x, y, epochs=9999999, batch_size=100, shuffle=True, use_multiprocessing=True)
        model.save(MODEL_PATH)
    except:
        model.save(MODEL_PATH)

def predict(mp3):
    sound = AudioSegment.from_mp3(mp3)
    (L, R) = prepare_y(sound)
    model = load_model(MODEL_PATH)
    LL = denormalize(flatten(model.predict(L)))
    RR = denormalize(flatten(model.predict(R)))
    samples = merge(LL, RR)
    samples = overlay(sound.get_array_of_samples(), samples)
    sound._spawn(samples).export("result-1000.mp3", format="mp3")

def analyze(path):
    sound = AudioSegment.from_mp3(path)
    plot_sample(sound)

prepare_model()
# info("Shawn Wasabi - Otter Pop (ft. Hollis).mp3")
# train("Shawn Wasabi - Otter Pop (ft. Hollis).mp3")
# predict("Shawn Wasabi - Otter Pop (ft. Hollis).mp3")
# predict("Trisha and Me - Do You Wanna Build A Snowman.mp3")
predict("result-1000.mp3")
predict("result-1000.mp3")
predict("result-1000.mp3")
predict("result-1000.mp3")
predict("result-1000.mp3")
predict("result-1000.mp3")
predict("result-1000.mp3")

# analyze("Shawn Wasabi - Otter Pop (ft. Hollis).mp3")
# analyze("result-1.mp3")
# analyze("Trisha and Me - Do You Wanna Build A Snowman.mp3")
# analyze("result.mp3")
analyze("result-1000.mp3")
