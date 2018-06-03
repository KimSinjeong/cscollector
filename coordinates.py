import numpy as np
import math
from euclid import Vector2, Vector3
import detection
import cv2

def align_to_robot(robot, corners, sodas, milks):
  robfront, robback = robot

  # calculate angle to rotate by
  robvec = Vector2(*robfront) - Vector2(*robback)
  theta = np.arctan2(robvec.y, robvec.x) - math.radians(90)
  theta = -theta

  # find upper-right and lower-left corners
  ca = corners[0]
  cb = corners[1]
  ur = max(ca[0], cb[0]), max(ca[1], cb[1])
  ll = min(ca[0], cb[0]), min(ca[1], cb[1])
  corners = [ur, ll]

  # rotate everything

  for i, soda in enumerate(sodas):
    v = Vector2(*soda) - Vector2(*robback)
    vrot = Vector3(v.x, v.y, 0).rotate_around(Vector3(0, 0, 1), theta)
    sodas[i] = vrot.x, vrot.y

  for i, milk in enumerate(milks):
    v = Vector2(*milk) - Vector2(*robback)
    vrot = Vector3(v.x, v.y, 0).rotate_around(Vector3(0, 0, 1), theta)
    milks[i] = vrot.x, vrot.y

  for i, corner in enumerate(corners):
    v = Vector2(*corner) - Vector2(*robback)
    vrot = Vector3(v.x, v.y, 0).rotate_around(Vector3(0, 0, 1), theta)
    corners[i] = vrot.x, vrot.y

  return corners, sodas, milks

def rela_coords():
  cap = cv2.VideoCapture(1)
  ret, image = cap.read()
  #cv2.imshow("Image", image)
  # cv2.waitKey(0)

  if not ret:
    raise Exception('Camera initialization failed.')

  robot, corners, sodas, milks, res_image = detection.detect(image)
  #cv2.imshow("detection", res_image)
  #cv2.waitKey(0)
  # if there are any problems return dummy map, so that robot will move
  if len(robot[0]) != 2 or len(robot[1]) != 2 or len(corners) != 2:
    dummy_corners = [[50, 50], [-50,-50]]
    dummy_sodas = []
    dummy_milks = [[0, 10]]
    #cv2.destroyAllWindows()
    #cap.release()
    return [len(dummy_milks), len(dummy_sodas)], np.array(dummy_corners + dummy_milks + dummy_sodas, dtype='int32')

  corners, sodas, milks = align_to_robot(robot, corners, sodas, milks)
  cv2.destroyAllWindows()
  cap.release()

  #print(np.array([*corners, *milks, *sodas], dtype='int32'))
  return [len(milks), len(sodas)], np.array([*corners, *milks, *sodas], dtype='int32')