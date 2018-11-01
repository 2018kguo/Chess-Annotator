import numpy as np
import cv2
import argparse
import glob
import matplotlib.pyplot as plt
import sys
import time
sys.path.append("../tensorflow-for-poets-2/scripts")
sys.path.append("../scripts")
import label_image
import chess
import copy
import imageio

#prevents the warning messages in console from tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def auto_canny(image, sigma):
	# compute the median of the single channel pixel intensities
	v = np.mean(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def checkfar(para, rho, diff):
	for storedrho, coord in para:
		if (storedrho - diff < rho < storedrho + diff):
			return False
	return True
	
def hough_lines(img,orig, fname):
	#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
	lines = cv2.HoughLines(img,1,6*np.pi/180,50)
	#parallel and perp line counters
	first = 0
	angle = 0
	para = []
	perp = []
	height,width = img.shape
	#print(width)
	#print(height)
	for i in range(len(lines)):
		for rho,theta in lines[i]:
			if(-0.2 < theta < 0.2) or (1.37 < theta < 1.77):
				if first == 0:
					if(-0.1 < theta < 0.1):
						angle = theta
					else:
						angle = (theta+np.pi/2)%np.pi
					first += 1
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))
				if ((angle - 0.2 < theta < angle + 0.2) and checkfar(para,rho,width * 0.03) and 0.05*width<rho<0.95*width and len(para) < 9):
					#print(len(para))
					#print(rho)
					para.append([rho,(x1+x2)/2])
					cv2.line(orig,(x1,y1),(x2,y2),(0,0,255),3)
				elif (((angle+np.pi/2)%np.pi - 0.2 < theta < (angle+np.pi/2)%np.pi + 0.2) and checkfar(perp,rho,width * 0.02) and .1*height<rho<0.8*height and len(perp) < 9):
					#print(rho)
					perp.append([rho,(y1+y2)/2])
					cv2.line(orig,(x1,y1),(x2,y2),(0,0,255),3)
	parac = []
	for rho,coord in para:
		parac.append(coord)
	perpc = []
	for rho,coord in perp:
		perpc.append(coord)
	squarec = []
	parac.sort()
	perpc.sort()
	if(len(perpc) == 8):
		perpc.append(perpc[7] + perpc[7] - perpc[6])
	if(len(parac) == 8):
		parac.append(parac[7] + parac[7] - parac[6])
	#cv2.imwrite("HOUGH" + fname,orig)
	if(len(perpc) == 9 and len(parac) == 9):
		#print(len(parac))
		#print(len(perpc))
		for i in range(9):
			for j in range(9):
				squarec.append([int(round(min(parac)+i*(float(max(parac))-float(min(parac)))/8.0)),int(round(min(perpc)+j*(float(max(perpc))-float(min(perpc)))/8.0))])			
		#for para,perp in squarec:
			#cv2.circle(orig,(para,perp), 5, (0,255,255), -1)
		
		return squarec
	
	else:
		
		print("len of parac and perpc")
		print(len(parac))
		print(len(perpc))
		return 0

def checkMatch(array, std, mean, thres, upperthres):
	#empty1 = 0, empty2 = 1, notempty1 = 2, notempty2 = 3
	if(array[0] == 0):
		return 0
	for i in range(len(array)):
		if(abs(array[i][0]-std) + abs(array[i][1]-mean) < thres):
			return 1
		elif(thres < abs(array[i][0]-std) + abs(array[i][1]-mean) < upperthres):
			return 2
	return 3

def readVideo(videoFile):
	vid = imageio.get_reader(videoFile,  'ffmpeg')
	fps = vid.get_meta_data()['fps']
	print(fps)
	duration = vid.get_meta_data()["duration"]
	print(vid.get_meta_data())
	print(duration)
	maxframes = int(fps*(duration-1))
	print(maxframes)
	fno = 0
	vidframes = []
	while(fno <= maxframes):
		image = vid.get_data(int(fno))
		#print(image.shape)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		vidframes.append(image)
		#cv2.imwrite('../Frames/chelsea-gray' + str(fno) + '.jpg', image)
		fno += int(fps)
        return vidframes	
#readVideo("../Videos/test.mp4")

def printBoard(array):
	print("-------------")
	tempstr = ""
	for i in range(len(array)):
		tempstr += array[i]
		if(i % 8 == 7):
			print(tempstr)
			tempstr = ""
	print("-------------")
			
def label(img,graph,label,coords,pnum):
	curp = 0
	testimgs = []
	testindexes = []
	testmeans = []
	crop20 = img[coords[20][1]+15:coords[30][1]-15,coords[0][0]+15:coords[10][0]-15]
	crop21 = img[coords[20][1]+15:coords[30][1]-15,coords[10][0]+15:coords[20][0]-15]
	matches = [[np.std(crop20),np.mean(crop20)],[np.std(crop21),np.mean(crop21)]]
	results = []
	e1 = False
	e2 = False
	for i in range(8):
		for j in range(8):
			if(curp < (64-pnum)):
				diff = abs(coords[10][1]-coords[20][1])
				imgcrop = img[coords[10*i][1]+diff/2:coords[10*i+10][1]-diff/2,coords[10*j][0]+diff/2:coords[10*j+10][0]-diff/2]
				std = np.std(imgcrop)
				mean = np.mean(imgcrop)
				#print([i,j,int(std),int(mean)])
				match = checkMatch(matches,int(std),int(mean),15,50)
				if(match==1):
					results.append(".")
				elif(match==3):
					diff = int(abs(coords[10*i][1]-coords[10*i+10][1])/4)
					centercrop = img[coords[10*i][1]+diff:coords[10*i+10][1]-diff,coords[10*j][0]+diff:coords[10*j+10][0]-diff]
					BLACK_MIN = np.array([0,0,0], np.uint8)
					BLACK_MAX = np.array([180,255,40], np.uint8)
					squarePixels = float(abs(coords[10][1]-coords[20][1]))*float(abs(coords[10][1]-coords[20][1]))
					dst = cv2.inRange(centercrop, BLACK_MIN, BLACK_MAX)
					dst = cv2.countNonZero(dst)/squarePixels
					if(dst > 0.1):
						results.append("b")
					else:
						results.append("w")
				#elif(mean < 50 and std > matches[0][0]+10):
				#	results.append("b")
				#elif(mean > 180 and std > matches[0][0]+10):
				#	results.append("w")
				else:
					#print(str(i) + " " + str(j))
					diff = int(abs(coords[10*i][1]-coords[10*i+10][1])/4)
					centercrop = img[coords[10*i][1]+diff:coords[10*i+10][1]-diff,coords[10*j][0]+diff:coords[10*j+10][0]-diff]
					BLACK_MIN = np.array([0,0,0], np.uint8)
					BLACK_MAX = np.array([180,255,40], np.uint8)
					squarePixels = float(abs(coords[10][1]-coords[20][1]))*float(abs(coords[10][1]-coords[20][1]))
					print(squarePixels)
					dst = cv2.inRange(centercrop, BLACK_MIN, BLACK_MAX)
					dst = cv2.countNonZero(dst)/squarePixels
					print([dst,i,j])
					testimgs.append(imgcrop)
					testindexes.append(len(results))
					testmeans.append(dst)
					#if(label_image.test(imgcrop,graph,label,128) == "empty"):
					#	results.append(".")
					#else:
					#	results.append("o")
					#	curp+=1
				#else:
				#	results.append("o")
					#curp+=1
			else:
				results.append(".")
	#start = time.time()
	imgresults = label_image.test2(testimgs,graph,label,224)
	#print("time load all images:"+ str(time.time()-start))
	#print(testindexes)
	#print(results)
	for i in range(len(imgresults)):
		if(imgresults[i] == "empty"):
			results.insert(testindexes[i]+i, ".")
		else:
			if(testmeans[i] > 0.1):
				results.insert(testindexes[i]+i, "b")
			else:
				results.insert(testindexes[i]+i, "w")
	print(imgresults)
	return results

def label2(img,coords):
	testimgs = []
	testindex = []
	board = []
	for i in range(8):
		for j in range(8):
			diff = abs(int(coords[10][1])-int(coords[20][1]))
			overallcrop = img[int(coords[10*i][1]):int(coords[10*i+10][1]),int(coords[10*j][0]):int(coords[10*j+10][0])]
			imgcrop = img[int(coords[10*i][1]+diff/4):int(coords[10*i+10][1]-diff/4),int(coords[10*j][0]+diff/4):int(coords[10*j+10][0]-diff/4)]
			imghsv = cv2.cvtColor(imgcrop, cv2.COLOR_BGR2HLS)
			squarePixels = float(abs(coords[10][1]-coords[20][1]))*float(abs(coords[10][1]-coords[20][1]))/4
			BLACK_MIN = np.array([0,0,0], np.uint8)
			BLACK_MAX = np.array([180,20,255], np.uint8)
			black = cv2.inRange(imghsv, BLACK_MIN, BLACK_MAX)
			black = cv2.countNonZero(black)/squarePixels 
			GREEN_MIN = np.array([30,0,0], np.uint8)
			GREEN_MAX = np.array([80,255,255], np.uint8)
			green = cv2.inRange(imghsv, GREEN_MIN, GREEN_MAX)
			green = cv2.countNonZero(green)/squarePixels
			GRAY_MIN = np.array([0,20,0], np.uint8)
			GRAY_MAX = np.array([180,50,255], np.uint8)
			white = cv2.inRange(imghsv, GRAY_MIN, GRAY_MAX)
			white = cv2.countNonZero(white)/squarePixels
			if(black > 0.25):
				result = "b"
				board.append(result)
			elif(green > 0.9):
				result = "."
				board.append(result)
			elif(green > 0.1):
				result = "w"
				board.append(result)
			else:
				hough = auto_canny(overallcrop, 0.1)
				cv2.imwrite("aaaa"+str(i)+str(j)+".jpg", hough)
				testimgs.append(overallcrop)
				testindex.append(8*i + j)
				result = "?"
				board.append(result)
			#print([i,j,"%.4f" % black,"%.2f" % green,"%.4f" % white, result])
	start = time.time()
	results = label_image.test2(testimgs,"../tf_files/chess_graph.pb","../tf_files/chess_labels.txt",224)
	for i in range(len(results)):
		#print(testindex[i])
		if(results[i] == "empty"):
			board[testindex[i]] = "."
		else:
			board[testindex[i]] = "w"
	#print("time load all images:"+ str(time.time()-start))		
	return board
def checkFrame(img,coords):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img = cv2.resize(img[coords[0][1]:coords[80][1],coords[0][0]:coords[80][0]],(100,100))
	cv2.imwrite("boardrange.jpg", img)
	BLUE_MIN = np.array([20,20,60], np.uint8)
	BLUE_MAX = np.array([50, 40,90], np.uint8)
	dst = cv2.inRange(img, BLUE_MIN, BLUE_MAX)
	cv2.imwrite("aaaaa.jpg", dst)
	#diff = int(abs(coords[0][1]-coords[10][1])/2)
	#oldside = img[coords[0][1]-diff:coords[0][1],coords[0][0]:coords[80][0]]
	#cv2.imwrite('../results/oldside.jpg', oldside)
	#oldside2 = img[coords[80][1]:coords[80][1]+diff,coords[0][0]:coords[80][0]]
	#cv2.imwrite('../results/oldside2.jpg', oldside2)
	#print(np.median(oldside))
	#print(np.median(oldside2))
		#if(np.sum(cv2.absdiff(oldside,newside)) > thres):
		#	outside = True
		#lower edge
		#oldside = cv2.resize(old[coords[80][1]:coords[80][1]+cdiff,coords[0][0]:coords[80][0]],(800,100))
		#cv2.imwrite('../results/oldside2.jpg', oldside)
		#newside = cv2.resize(new[coords2[80][1]:coords2[80][1]+cdiff,coords2[0][0]:coords2[80][0]],(800,100))
		#if(np.sum(cv2.absdiff(oldside,newside)) > thres):
		#	outside = True
	#if(outside == True):
		#return 0
	#if(outside == True):
		#results = []
		#for i in range(8):
			#for j in range(8):
				#oldcrop = cv2.resize(old[coords[10*i][1]:coords[10*i+10][1],coords[10*j][0]:coords[10*j+10][0]],(100,100))
				#newcrop = cv2.resize(new[coords2[10*i][1]:coords2[10*i+10][1],coords2[10*j][0]:coords2[10*j+10][0]],(100,100))
				#cv2.imwrite("../results/imgr"+str(i)+"c"+str(j)+".jpg", cv2.subtract(oldcrop,newcrop))
				#diff = np.median(cv2.subtract(oldcrop,newcrop))
				###if(diff < thres):
				###	results.append("s")
				###else:
				###	results.append("d")
				#results.append(diff)
		#return results

def checkFrameTest(img, coords):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img = cv2.resize(img[coords[0][1]:coords[80][1],coords[0][0]:coords[80][0]],(300,300))
	BLUE_MIN = np.array([0,25,25], np.uint8)
	BLUE_MAX = np.array([50, 150,160], np.uint8)
	dst = cv2.inRange(img, BLUE_MIN, BLUE_MAX)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	dst = cv2.erode(dst, kernel, iterations = 2)
	dst = cv2.dilate(dst, kernel, iterations = 2)
	dst = cv2.GaussianBlur(dst, (3, 3), 0)
	cv2.imwrite("aaaaa.jpg", dst)
	if(cv2.countNonZero(dst) > 1500):
		return False
	return True
	
def readVideo(fileName):
	videoFile = "capture.avi"
	imagesFolder = "../Frames"
	cap = cv2.VideoCapture(videoFile)
	frameRate = cap.get(5) #frame rate
	while(cap.isOpened()):
		frameId = cap.get(1) #current frame number
		ret, frame = cap.read()
		if (ret != True):
			break
		if (frameId % math.floor(frameRate) == 0):
			filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
			cv2.imwrite(filename, frame)
			#frames = []
			#frames.append(frame)
	cap.release()
	#return frames
	
def getBoardStates(directory):
	boardstates = []
	coords = []
	frames = []
	for filename in os.listdir(directory):
		if filename.endswith(".jpg"): 
			print(filename)
			start = time.time()
			img = cv2.imread(directory + "/"+filename)
			frames.append(img)
			canny = auto_canny(img, 0.33)
			#hough = cv2.imread('canny.jpg')
			coords = hough_lines(canny,img,filename)
			if(coords != 0):
				#checkFrame(img, coords)
				coords.append(coords)
				state = label2(img,coords)
				#printBoard(state)
				boardstates.append(state)
				print("time to read board:"+ str(time.time()-start))
	#for frame in directory:
		#start = time.time()
		#canny = auto_canny(frame)
		##hough = cv2.imread('canny.jpg')
		#coords = hough_lines(canny,img)
		#if(coords != 0):
			#checkFrame(img, coords)
			#boardstates.append(label(img,"../tf_files/chess_graph.pb","../tf_files/chess_labels.txt",coords,16))
			#print("time to read board:"+ str(time.time()-start))
	return boardstates, frames, coords

def getBoardStates2(vidframes):
	boardstates = []
	coords = []
	for frame in vidframes:
		start = time.time()
		canny = auto_canny(frame, 0.33)
		coords = hough_lines(canny,frame,filename)
		if(coords != 0):
			#checkFrame(img, coords)
			coords.append(coords)
			state = label2(img,coords)
			#printBoard(state)
			boardstates.append(state)
			print("time to read board:"+ str(time.time()-start))
	return boardstates, vidframes, coords
	
def getMoves(boardstates, frames, coords):
	result = ""
	board = chess.Board()
	index = 0
	while(board.is_game_over() == False and index < len(boardstates)-1):
                uci = getSingleMove(boardstates[index],boardstates[index+1], frames[index], coords[index], board)
                if(uci != 0):
                        move = chess.Move.from_uci(uci)
                        result += board.san(move)
                        result += "\n"
                        board.push(move)
                        print(board)
		index += 1
	return result
		
def getSingleMove(oldb, newb, frame, coords, board):
	#printBoard(oldb)
	#printBoard(newb)
	differences = []
	for k in range(len(oldb)):
		if(oldb[k] != newb[k]):
			if(newb[k] == "."):
				differences.insert(0,[k,oldb[k],newb[k]])
			else:
				differences.append([k,oldb[k],newb[k]])
	#move
	print(differences)
	if(len(differences) == 2):
		#promotion
		if(board.piece_at(64-differences[0][0]).upper()=="P" and ((differences[0][0]/8 == 6 and differences[1][0] / 8 == 7) or (differences[0][0]/8 == 1 and differences[1][0] / 8 == 0))):
			k = differences[0][1]
			imgcrop = frame[coords[10*(k/8)][1]+15:coords[10*(k/8)+10][1]-15,coords[10*(k%8)][0]+15:coords[10*(k%8)+10][0]-15]
			res = label_image.test2(imgcrop,"../tf_files/chess_graph2.pb","../tf_files/chess_labels2.txt",224)
			if(res == "bq" or res == "wq" or res == "bk" or res == "wk" ):
				print(sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0]))
				return sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0]) + "q"
			elif(res == "wn" or res == "bn"):
				print(sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0]))
				return sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0]) + "n"
			elif(res == "wr" or res == "br"):
				print(sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0]))
				return sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0]) + "r"
			elif(res == "wb" or res == "bb"):
				print(sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0]))
				return sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0]) + "b"
		############
		else:
			print(sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0]))
			return sqFromIndex(differences[0][0]) + sqFromIndex(differences[1][0])
	#en passant
        if(len(differences) == 3):
                for diff in differences:
                        if(diff[0]/8==2 or diff[0]/8==5):
                                capturecolor = diff[2]
                                newsq = sqFromIndex(diff[0])
                                break
                for diff in differences: 
                        if(diff[0]/8==3 or diff[0]/8==4 and diff[2] == capturecolor):
                                oldsq = sqFromIndex(diff[0])
                                break
                return oldsq+newsq
        #castle
	if(len(differences) == 4):	
		if(differences[0][0] == 63 or differences[1][0] == 63):
			return "e1c1"
		else:
			return "e1g1"
	else:
		return 0	
		
def sqFromIndex(index):
	sqstr = ""
	cols = ["a","b","c","d","e","f","g","h"]
	sqstr += cols[index%8]
	#may have to change this later
	sqstr += str(8 - int(index/8))
	return sqstr
	
#def main(video_name):##for para,perp in coords2:

	#boardstates = readVideo()
	#return getMoves(boardstates) write("gray.jpg", gray)
##cv2.imwrite('norm.jpg',norm_image)
##cv2.imwrite("canny.jpg", canny)

#hough = cv2.imread('canny.jpg')
#coords = hough_lines(hough,1)
##clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
##lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
##l, a, b = cv2.split(lab)  # split on 3 different channels
##l2 = clahe.apply(l)  # apply CLAHE to the L-channel
##lab = cv2.merge((l2,a,b))  # merge channels
##img2 = cv2.bilateralFilter(img,9,75,75)  # convert from LAB to BGR
##img2 = cv2.imread('../Photos/f2.jpg')
##small = cv2.resize(img2, (0,0), fx=0.5, fy=0.5) 
##norm_image = cv2.normalize(small,cv2.CV_32F, 80, 175, norm_type=cv2.NORM_MINMAX)
##canny2 = auto_canny(img2)
##cv2.imwrite('norm.jpg',norm_image)
##cv2.imwrite("canny2.jpg", canny2)
##hough2 = cv2.imread('canny2.jpg')
##coords2 = hough_lines(hough2,1)
##for para,perp in coords2:
##	cv2.circle(img2,(para,perp), 4, (0,255,255), -1)
##cv2.imwrite("../results/overlay.jpg", img2)
##print(compareFrame(img, img2, coords, coords2))
#start = time.time()
#res = label(img,"../tf_files/chess_graph3.pb","../tf_files/chess_labels3.txt",coords,16)
#printBoard(res)
#end = time.time()
#print("time:"+ str(end-start))
#for para,perp in coords:
	#cv2.circle(img,(para,perp), 4, (0,255,255), -1)
#cv2.imwrite("../results/overlay.jpg", img)

##states, frames, coords = getBoardStates("../Photos2")
##print(getMoves(states, frames, coords))

#img = cv2.imread("../Frames/f1.jpg")
#canny = auto_canny(img)
#coords = hough_lines(canny,img)
#if(coords != 0):
	#checkFrameTest(img,coords)

#for filename in os.listdir("../Photos2"):
	#if filename.endswith(".jpg"): 
		#print(filename)
		#start = time.time()
		#img = cv2.imread("../Photos2" + "/"+filename)
		#canny = auto_canny(img)
		##hough = cv2.imread('canny.jpg')
		#coords = hough_lines(canny,img,filename)
		#if(coords != 0):
			#label2(img,coords)

states, frames, coords = getBoardStates("../Photos2")
print(getMoves(states,frames,coords))

