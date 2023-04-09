import cv2

# Specify the paths for the 2 files
protoFile = r'C:\Users\hwojc\Desktop\Diplomka\OpenPose\repo\openpose\models\pose\mpi\pose_deploy_linevec_faster_4_stages.prototxt'
weightsFile = r'C:\Users\hwojc\Desktop\Diplomka\OpenPose\repo\openpose\models\pose\mpi\pose_iter_160000.caffemodel'

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Read image
imgPath = r'C:\Users\hwojc\Desktop\Microsoft dataset\Angry\Picture2.png'
frame = cv2.imread(imgPath)
  
# Specify the input image dimensions
inWidth = 1139
inHeight = 2475
  
# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(
    frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
  
# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

output = net.forward()

H = output.shape[2]
W = output.shape[3]
# Empty list to store the detected keypoints
points = []
threshold = 10
for i in range(len(output)):
	# confidence map of corresponding body's part.
	probMap = output[0, i, :, :]

	# Find global maxima of the probMap.
	minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

	# Scale the point to fit on the original image
	x = (inWidth * point[0]) / W
	y = (inHeight * point[1]) / H

    
	if prob > threshold:
		cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255),
				thickness=-1, lineType=cv2.FILLED)
		cv2.putText(frame, "{}".format(i), (int(x), int(
			y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)

		# Add the point to the list if the probability is greater than the threshold
		points.append((int(x), int(y)))
	else:
		points.append(None)

cv2.imshow("Output-Keypoints", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
