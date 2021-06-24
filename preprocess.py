# data loader
import cv2
import os

def loadSingleVideo(path, fileName, outPath):
    # make sure that the fileName can be found under the "path" variable
    # the function will create a folder with name "fileName" under the "outPath" directory and put all the images
    # under this directory. The images are all the frames of the video
    outDir = outPath + fileName[:-4]+"/"
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    cap = cv2.VideoCapture(os.path.join(path, fileName))
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(outDir+ str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

def convertVideosToJPGs(path, fileNames, outPath):
    for file in fileNames:
        loadSingleVideo(path, file, outPath)