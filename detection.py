import numpy as np
import argparse
import cv2
import imutils

# GREEN
# lower = (0,    83-5, 128-5)
# upper = (255, 103+5, 143+5)

# YELLOW
# lower = (0,   -15+128, 20+128)
# upper = (255,   5+128, 40+128)

# RED
# lower = (0,   40+128, 20+128)
# upper = (255, 75+128, 50+128)

# BLUE
# lower = (0,   -20+128, -55+128)
# upper = (255,   5+128, -20+128)

# (BAD) ORANGE
# lower = (0,    6+128, 10+128)
# upper = (255, 35+128, 40+128)

# CYAN
# upper = (0,   -15+128, -5+128)
# lower = (255,   0+128,  5+128)

def detect(image):
  image = image.copy()
  robfront = ()
  robback = ()
  corners = []
  sodas = []
  milks = []
  
  # thresholding
  bounds = {
    "green":  [(0,    83-5, 128-5),     (255, 103+5, 143+5)],
    "yellow": [(0,   -15+128, 20+128),  (255,   5+128, 40+128)],
    "red":    [(0,   40+128, 20+128),   (255,  75+128, 50+128)],
    #"red":    [(0,   20 + 128, 10 + 128),   (255,  75+128, 50+128)],
    "blue":   [(0,   -20+128, -55+128), (255,   5+128, -20+128)],
    # "cyan":   [(50,   -15+128, -5+128),  (200,   0+128,  5+128)],
  }

  image = cv2.GaussianBlur(image, (5, 5), 0)
  lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  
  for (i, (color, colorbound)) in enumerate(bounds.items()):
    lower = colorbound[0]
    upper = colorbound[1]
    mask = cv2.inRange(lab_image, lower, upper)
    # image = cv2.bitwise_and(image, image, mask = mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for c in cnts:
      M = cv2.moments(c)

      if M["m00"] == 0:
        continue

      # ignore small fragmentations
      # if M["m00"] < 300:
      #   print('area <300 ignored')
      #   continue

      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])

      # Detect shape
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.1 * peri, True)
      sides = len(approx)

      label = '{} {}-gon'.format(color, sides)

      # Draw shape
      cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
      # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
      cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
      cv2.putText(image, label, (cX+5, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

      if color == 'red' and sides == 3:
        corners.append((cX, cY))
      elif color == 'red' and sides == 4:
        robfront = (cX, cY)
      elif color == 'blue':
        robback = (cX, cY)
      elif color == 'green':
        milks.append((cX, cY))
      elif color == 'yellow':
        sodas.append((cX, cY))

  # return image
  return [robfront, robback], corners, sodas, milks, image

def main():
  cap = cv2.VideoCapture(1)

  while True:
    # Read image from frame
    ret, image = cap.read()
    if not ret:
      print("Couldn't read capture data")
      break

    _, _, _, _, res_image = detect(image)

    # Show frame
    cv2.imshow("detection", res_image)

    # Press q to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break

if __name__ == '__main__':
  main()