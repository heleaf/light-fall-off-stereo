
import numpy as np
import skimage
from cp_hw2 import lRGB2XYZ
import matplotlib.pyplot as plt
import math

def local_method(delta_r, I, Iprime):
	Iprime = np.clip(Iprime, 0.01, np.max(Iprime))
	denom = np.sqrt(I/Iprime) - 1
	denom += 0.0001 # prevent division by 0
	return delta_r / denom

def display_surface(Z):
	H, W = np.shape(Z)
	x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(x, y, Z, c=Z, cmap='gray', s=1)
	ax.set_xlim3d([0, W])
	ax.set_ylim3d([0, H])
	plt.axis('off')
	plt.show()

def linearize_rgb(img): # sRGB inverse gamma enc operator
	threshold = 0.0404482
	less_mask = np.ma.getmask(
		np.ma.masked_where(img <= threshold, img)).astype(np.float64)
	more_mask = 1.0 - less_mask 
	less = img * less_mask
	more = img * more_mask
	less /= 12.92
	more += 0.055
	more /= 1.055
	more = np.power(more, 2.4)
	linear = less + more
	return linear

# lambd = smoothing factor weight 
def global_method(imgPath, rowBounds, colBounds, downsample, 
lambd, saveName, imgstart=0, imgend=16):
	out = None
	sqrt_sum_img = None
	dist_scaled_sqrt_sum_img = None
	mask = None
	ogShape = None

	for i in range(imgstart*10, imgend*10, 10):
		print("adding img", i, "to solver")
		img = skimage.io.imread(f'{imgPath}{i}.jpg')[
			rowBounds[0]:rowBounds[1]:downsample, 
			colBounds[0]:colBounds[1]:downsample, :]
		xyz = lRGB2XYZ(linearize_rgb(img))
		lum = xyz[:, :, 1]

		if i == imgstart*10:
			# collect boundary mask-- use later to set gradient to 0
			# on boundary pixels in conjugate gradient descent 
			mask = np.ones_like(lum)
			mask = np.pad(mask, (1,1))

		lum = np.pad(lum, (1,1))

		if i == imgstart*10:
			ogShape = np.shape(lum)

		if i == imgstart*10:
			out = np.expand_dims(lum.flatten(), axis=0)
			print(np.shape(out))
			sqrt_sum_img = np.sqrt(lum)
			dist_scaled_sqrt_sum_img = (sqrt_sum_img.flatten()) * i
		else:
			row = np.expand_dims(lum.flatten(), axis=0)
			print(np.shape(row))
			out = np.concatenate((out, row), axis=0)
			toadd = np.sqrt(lum)
			sqrt_sum_img += toadd
			dist_scaled_sqrt_sum_img += (toadd.flatten() * i)

	(rows, cols) = np.shape(out)
	out = np.sqrt(out) 
	sqrt_sum_img = sqrt_sum_img.flatten()
	A = np.zeros((cols,cols))
	b = np.zeros((cols, 1))

	ogRows, ogCols = ogShape[0], ogShape[1]

	for i in range(cols):
		print('Processing pixel', i)
		A_total = 0
		b_total = 0
		for img in range(0, imgend-imgstart):
			sqrt_img_pixel = out[img, i]
			a_img = sqrt_img_pixel - (sqrt_sum_img[i]/rows)
			A_total += 2*(a_img**2)
			b_img = (sqrt_img_pixel*img*10) - (dist_scaled_sqrt_sum_img[i]/rows)
			b_total += -2*a_img*b_img

		#### minimizing variance
		A[i, i] += (1-lambd) * A_total
		b[i, 0] +=  (1-lambd) * b_total

		if lambd == 0: continue

		#### smoothing equations

		# ogRows * r + c = i
		# # (r,c) = (y, x)
		# i = ogRows * r + c
		# r = i // ogRows 
		# c = (i - (ogRows * r))
		curr_row = i // ogRows
		curr_col = i - (ogRows * curr_row)
		
		# u_xy^2 
		A[i, i] += 2*lambd*4
		if curr_col == 0:
			# only set some of them.
			if i+1 < cols:
				# i-1 is out of bounds
				A[i, i+1] += 2*lambd * -2
				A[i+1, i] += 2*lambd * -2
				A[i+1, i+1] += 2*lambd * 1
		elif curr_col == ogCols - 1:
			# don't set the +1's
			if i-1 >= 0: 
				# i+1 is out of bounds
				A[i-1, i-1] += 2*lambd * 1
				A[i-1, i] += 2*lambd * -2
				A[i, i-1] += 2*lambd * -2
		else:
			if i-1 >= 0 and i+1 < cols:
				# everything in bounds 
				A[i-1, i-1] += 2*lambd * 1
				A[i-1, i] += 2*lambd * -2
				A[i-1, i+1] += 2*lambd * 1
				A[i, i-1] += 2*lambd * -2
				A[i, i+1] += 2*lambd * -2
				A[i+1, i-1] += 2*lambd * 1
				A[i+1, i] += 2*lambd * -2
				A[i+1, i+1] += 2*lambd * 1
			elif i-1 >= 0: 
				# i+1 is out of bounds
				A[i-1, i-1] += 2*lambd * 1
				A[i-1, i] += 2*lambd * -2
				A[i, i-1] += 2*lambd * -2
			elif i+1 < cols:
				# i-1 is out of bounds
				A[i, i+1] += 2*lambd * -2
				A[i+1, i] += 2*lambd * -2
				A[i+1, i+1] += 2*lambd * 1

		# v_xy^2
		prev_row_idx_at_curr_col = (ogRows * (curr_row - 1)) + curr_col
		next_row_idx_at_curr_col = (ogRows * (curr_row + 1)) + curr_col
		A[i, i] += 2*lambd*4
		if curr_row == 0:
			# don't set any prev_row values
			A[next_row_idx_at_curr_col, next_row_idx_at_curr_col] += 2*lambd
			A[i, next_row_idx_at_curr_col] += 2*lambd*-2
			A[next_row_idx_at_curr_col, i] += 2*lambd*-2
		elif curr_row == ogRows - 1:
			# don't set any next_row values
			A[prev_row_idx_at_curr_col, i] += 2*lambd*-2
			A[i, prev_row_idx_at_curr_col] += 2*lambd*-2
			A[prev_row_idx_at_curr_col, prev_row_idx_at_curr_col] += 2*lambd
		else:
			if prev_row_idx_at_curr_col >= 0 and next_row_idx_at_curr_col < cols:
				A[next_row_idx_at_curr_col, next_row_idx_at_curr_col] += 2*lambd
				A[i, next_row_idx_at_curr_col] += 2*lambd*-2
				A[prev_row_idx_at_curr_col, next_row_idx_at_curr_col] += 2*lambd
				A[next_row_idx_at_curr_col, prev_row_idx_at_curr_col] += 2*lambd
				A[next_row_idx_at_curr_col, i] += 2*lambd*-2
				A[prev_row_idx_at_curr_col, i] += 2*lambd*-2
				A[i, prev_row_idx_at_curr_col] += 2*lambd*-2
				A[prev_row_idx_at_curr_col, prev_row_idx_at_curr_col] += 2*lambd
			elif prev_row_idx_at_curr_col < 0:
				A[next_row_idx_at_curr_col, next_row_idx_at_curr_col] += 2*lambd
				A[i, next_row_idx_at_curr_col] += 2*lambd*-2
				A[next_row_idx_at_curr_col, i] += 2*lambd*-2
			elif next_row_idx_at_curr_col >= cols:
				A[prev_row_idx_at_curr_col, i] += 2*lambd*-2
				A[i, prev_row_idx_at_curr_col] += 2*lambd*-2
				A[prev_row_idx_at_curr_col, prev_row_idx_at_curr_col] += 2*lambd

	initialSol = np.zeros_like(b)
	flat_mask = np.expand_dims(mask.flatten(), axis=0).T
	final = conjugateGradientDescent(b, A, 4000, initialSol, flat_mask, 0.001)
	reshaped = np.reshape(final, ogShape)
	plt.imshow(reshaped, cmap='gray')
	plt.show()
	skimage.io.imsave(saveName, reshaped)
	display_surface(reshaped)

def conjugateGradientDescent(b, A, iterN, x, boundaryMask, eps):
	r = boundaryMask * (b - (A @ x)) 
	d = r
	print("conjugate gradient descending...")
	delta_new = np.sum(r * d) 
	n = 0
	while math.sqrt(delta_new) > eps and n < iterN:
		print('iter', n)
		q = A @ d 
		mu = delta_new / np.sum(d*q) 
		x += (boundaryMask * (mu * d)) 
		r = boundaryMask * (r - (mu * q)) 
		delta_old = delta_new
		delta_new = np.sum(r*r)  
		beta = delta_new / delta_old
		d = r + (beta * d)
		n = n + 1
	print('ending resid', math.sqrt(delta_new))
	return x

def main():
	i1 = skimage.io.imread(f'aveeno/dist30.jpg') [680:780, 2840:2940]
	i2 = skimage.io.imread(f'aveeno/dist40.jpg') [680:780, 2840:2940]
	xyz1 = lRGB2XYZ(linearize_rgb(i1))
	xyz2 = lRGB2XYZ(linearize_rgb(i2))
	luminanceChannel1 = xyz1[:, :, 1]
	luminanceChannel2 = xyz2[:, :, 1]
	test = local_method(10, luminanceChannel1, luminanceChannel2)
	plt.imshow(test)
	skimage.io.imsave('local_method.PNG', test)
	plt.show()

	global_method('aveeno/dist', (680, 780), (2840, 2940), 
		1, 0.15, 'global_method.PNG')

main()