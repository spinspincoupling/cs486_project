'''
This script handles the training process.
'''
import preprocess
import parameters
def train():
    print("Start training...")


def main():

    preprocess.convertVideosToJPGs("/Users/bzjesus/Documents/CS486/diving", parameters.videoNames,"./pics/")

    train()


if __name__ == '__main__':
    main()
