class DecisionMaker():
  def __init__(self):
    self.right_seat_person = None
    self.left_seat_person = None

  def set_right_seat_person(self, right_seat_person):
    self.right_seat_person = right_seat_person
  
  def set_left_seat_person(self, left_seat_person):
    self.left_seat_person = left_seat_person
  
  def get_right_seat_person(self):
    return self.right_seat_person
  
  def get_left_seat_person(self):
    return self.left_seat_person
    