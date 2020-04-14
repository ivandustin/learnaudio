from pydub import AudioSegment
import matplotlib.pyplot as plt
import random as rnd
import numpy
import neural
from keras.models import load_model

MIN = -32768
MAX = 32767

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
		r = rnd.randint(-100, 100)
		n = v + r
		if (v > 0 and n < 32768 and n > -32768):
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
	input_size = 100
	(L, R) = bands(sound)
	n = int(len(L) / input_size)
	x = []
	y = []
	for i in range(0, n):
		a = normalize(noise(L[i*input_size:(i*input_size)+input_size]))
		b = normalize(L[i*input_size:(i*input_size)+input_size])
		x.append(numpy.array(a + [1]))
		y.append(numpy.array(b))
	return (numpy.array(x), numpy.array(y))

def prepare_y(sound):
	input_size = 100
	(L, R) = bands(sound)
	n = int(len(L) / input_size)
	l = []
	r = []
	for i in range(0, n):
		a = normalize(L[i*input_size:(i*input_size)+input_size])
		b = normalize(R[i*input_size:(i*input_size)+input_size])
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

def info(sound):
	print("channels", sound.channels)
	print("frame rate", sound.frame_rate)
	print("sample width", sound.sample_width)
	print("max amplitude", sound.max)
	print("duration(minutes)", len(sound) / 1000 / 60)
	print("dBFS", sound.dBFS)
	samples = sound.get_array_of_samples()
	print("sample length", len(samples))
	print("sample", sample(sound))

def train(sound):
	(x, y) = prepare_x_y(sound)
	# model = neural.network(101, [300, 250, 200, 150], 100)
	model = load_model("model.h5")
	try:
		model.fit(x, y, epochs=10000, initial_epoch=344, batch_size=100, shuffle=True, use_multiprocessing=True)
		model.save("model.h5")
	except:
		model.save("model.h5")

def predict(sound):
	(L, R) = prepare_y(sound)
	model = load_model("model.h5")
	LL = denormalize(flatten(model.predict(L)))
	RR = denormalize(flatten(model.predict(R)))
	samples = merge(LL, RR)
	samples = overlay(sound.get_array_of_samples(), samples)
	sound._spawn(samples).export("result.mp3", format="mp3")

def analyze(path):
	sound = AudioSegment.from_mp3(path)
	plot_sample(sound)

# sound = AudioSegment.from_mp3("Shawn Wasabi - Otter Pop (ft. Hollis).mp3")
# sound = AudioSegment.from_mp3("result.mp3")
sound = AudioSegment.from_mp3("Trisha and Me - Do You Wanna Build A Snowman.mp3")
# info(sound)
# train(sound)
# predict(sound)

# analyze("Shawn Wasabi - Otter Pop (ft. Hollis).mp3")
# analyze("result-1.mp3")
# analyze("Trisha and Me - Do You Wanna Build A Snowman.mp3")
# analyze("result.mp3")
