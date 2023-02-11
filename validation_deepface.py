#Comment
import cv2
import os
from deepface import DeepFace
path = r'C:\Users\hwojc\Desktop\Diplomka\AffectNet\Validation\angry'
correctEmotion = 'angry'

totalClassification = 0
angryClassification = 0
disgustClassification = 0
fearClassification = 0
happyClassification = 0
neutralClassification = 0
sadClassification = 0
surpsiseClassification = 0
correctClassification = 0

for images in os.listdir(path):
    if(images.endswith(".jpg")):
         path2 = os.path.join(path,images)
         print(path2)
         img = cv2.imread(path2)
         result = DeepFace.analyze(img, actions = ['emotion'], enforce_detection= False)

         dominantEmotion = result['dominant_emotion']
         dominantEmotionPct = result['emotion'][result['dominant_emotion']]
         print(dominantEmotion, ":",  dominantEmotionPct, "%")
         
         secondDominantEmotion = result['second_dominant_emotion']
         secondDominantEmotionPct =  result['emotion'][result['second_dominant_emotion']]
         print(secondDominantEmotion, ":",  secondDominantEmotionPct, "%")

         totalClassification += 1
        
         if(secondDominantEmotionPct > 10):
            if((dominantEmotion == correctEmotion) or (secondDominantEmotion == correctEmotion)):
                correctClassification += 1
            else: 
             match dominantEmotion:
                case 'angry':
                    angryClassification += 1
                case 'disgust':
                    disgustClassification += 1
                case 'fear':
                    fearClassification += 1
                case 'happy':
                    happyClassification += 1
                case 'neutral':
                    neutralClassification += 1
                case 'sad':
                    sadClassification += 1
                case 'surprise':
                    surpsiseClassification += 1
         else:
            if(dominantEmotion == correctEmotion):
                correctClassification += 1
            else: 
             match dominantEmotion:
                case 'angry':
                    angryClassification += 1
                case 'disgust':
                    disgustClassification += 1
                case 'fear':
                    fearClassification += 1
                case 'happy':
                    happyClassification += 1
                case 'neutral':
                    neutralClassification += 1
                case 'sad':
                    sadClassification += 1
                case 'surprise':
                    surpsiseClassification += 1



print('Angry:', angryClassification, '\n')
print('Disgust:', disgustClassification, '\n')
print('Fear:', fearClassification, '\n')
print('Happy:', happyClassification, '\n')
print('Neutral:', neutralClassification, '\n')
print('Sad:', sadClassification, '\n')
print('Surprise:', surpsiseClassification, '\n')
print('Correct classification:', correctClassification, '\n')
print('Total images:', totalClassification, '\n')

cv2.destroyAllWindows()


