import cv2
import numpy as np

def extract_frame(filepath, frame_no):
	'''
	Return an image in the nth frame of a video sequence

	input:
		filepath: the relative or absolute path of the video file
		frame_no: The nth frame we want to extract

	return:
		a cv2 image matrix of the nth frame of the video sequence
	'''
    cap = cv2.VideoCapture(filepath) #video is the video being called
    cap.set(1,frame_no); # Where frame_no is the frame you want
    ret, frame = cap.read() # Read the frame
    return frame