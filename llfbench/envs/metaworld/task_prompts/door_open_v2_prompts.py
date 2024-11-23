hp_move_to_door_feedback = (
    "Because you haven't reached the door handle yet, you should now moving to the door handle. Because the door handle is at {door}, doing the action {action} does help achieving the door handle. Thus,",
)

hp_open_door_feedback = (
    "Because you have already reached the door at {door}, you should now open it. Doing the action {action} does help opening the door. Thus,",    
)

hn_move_to_door_feedback = (
    "Because you haven't reached the door handle yet, you should now reach out to the door handle. But executing the action {action} is not helping to achieve the door handle at {door}. Thus,",
)

hn_closing_door_open_door_feedback = (
    "Because you have already reached the door at {door}, you should now open the door. But executing the action {action} is not helping to open it. In fact, you're closing the door now. Thus,",    
)

hn_losing_grip_open_door_feedback = (
    "You should now open the door at {door}. But executing the action {action} is not helping to open it. In fact, you're losing grip to the door handle. Thus,",    
)

# No tensor in description
hp_move_to_door_feedback_no_tensor = (
    "Because you haven't reached the door handle yet, you should now moving to the door handle. Based on the hand_pos and door_pos, doing the action does help achieving the door handle. Thus,",
)

hp_open_door_feedback_no_tensor = (
    "Because you have already reached the door at door_pos, you should now open it. According to the hand_pos and door_pos, Doing the action does help opening the door. Thus,",    
)

hn_move_to_door_feedback_no_tensor = (
    "Because you haven't reached the door handle yet, you should now reach out to the door handle. But executing the action is not helping to achieve the door handle according to the door_pos and hand_pos. Thus,",
)

hn_closing_door_open_door_feedback_no_tensor = (
    "Because you have already reached the door at door_pos, you should now open the door. But executing the action is not helping to open it based on the door_pos and the hand_pos. In fact, you're closing the door now. Thus,",    
)

hn_losing_grip_open_door_feedback_no_tensor = (
    "You should now open the door at door_pos. But executing the action is not helping to open it. In fact, considering the door_pos and hand_pos, you're losing grip to the door handle. Thus,",    
)
