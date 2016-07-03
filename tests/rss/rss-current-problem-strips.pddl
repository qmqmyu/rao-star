(define (problem Planning_Problem)
    (:domain RSS_Project_Rover_Scenario)
    (:objects
      rover1 - Rover
      l1 - Location
      l2 - Location
      l3 - Location
      l4 - Location
      l5 - Location
      unknown - Location
      pic_req1 - PictureRequest
      pic_req2 - PictureRequest
      pic_req3 - PictureRequest
      rock_req1 - RockSampleRequest
      rock_req2 - RockSampleRequest
    )
    (:init
        (at rover1 l1)
        (has_rock l2)
        (has_rock l4)
        (has_rock l5)
        (can_go_to l1)
        (can_go_to l2)
        (can_go_to l3)
        (can_go_to l4)
        (can_go_to l5)
        (idle rover1)
        (location pic_req1 l3)
        (location pic_req2 l2)
        (location pic_req3 l5)
        (mastcam_ready rover1)
        (hazcam_ready rover1)
        (orbiter_communication_available l2)
        (orbiter_communication_available l4)
        (pending pic_req1)
        (pending pic_req2)
        (pending pic_req3)
        (pending rock_req1)
        (pending rock_req2)
    )
    (:goal
      (and
        (transmitted pic_req1)
        (transmitted pic_req2)
        (transmitted pic_req3)
        (transmitted rock_req1)
        (transmitted rock_req2)
      )
    )
)
