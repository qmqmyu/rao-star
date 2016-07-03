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
      (mastcam_issue rover1)
      (at rover1 l3)
      (orbiter_communication_available l4)
      (orbiter_communication_available l2)
      (idle rover1)
      (hazcam_ready rover1)
      (has_rock l2)
      (has_rock l5)
      (has_rock l4)
      (location pic_req1 l3)
      (pending pic_req1)
      (pending rock_req1)
      (location pic_req2 l2)
      (pending pic_req2)
      (location pic_req3 l5)
      (pending pic_req3)
      (pending rock_req2)
      (communication_available l2)
      (communication_available l4)
            (orbiter_communication_available l2)
            (orbiter_communication_available l4)
    )
    (:goal
      (and
        (transmitted pic_req1)
      )
    )
)

