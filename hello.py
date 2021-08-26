from ohbot import ohbot
import sys


ohbot.reset()
ohbot.setVoice('Karen')
ohbot.setSpeechSpeed(190)
text = input("Question:\n")
ohbot.say(text)
ohbot.reset()
ohbot.close()
sys.exit(0)


#ohbot.setVoice('alto')
ohbot.say("Oh, no")
ohbot.move(ohbot.HEADTURN,3)
ohbot.wait(0.5)
ohbot.move(ohbot.HEADTURN,7)
ohbot.wait(0.5)
ohbot.move(ohbot.HEADTURN,5)

ohbot.say("Oh, yes!")
for i in range(2):
  ohbot.move(ohbot.HEADNOD,0)
  ohbot.wait(0.5)
  ohbot.move(ohbot.HEADNOD,5)
  ohbot.wait(0.5)

ohbot.move(ohbot.EYETILT,2)
ohbot.move(ohbot.EYETURN,2)
ohbot.wait(3)
ohbot.reset()
ohbot.close()
