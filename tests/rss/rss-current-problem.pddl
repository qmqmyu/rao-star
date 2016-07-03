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
      (at rover1 l3)
      (transmitted pic_req3)
      (transmitted rock_req1)
      (done rock_req1)
      (surveyed l4)
      (surveyed l5)
      (done pic_req3)
      (transmitted rock_req2)
      (done rock_req2)
      (surveyed l2)
      (transmitted pic_req2)
      (done pic_req2)
      (orbiter_communication_available l4)
      (orbiter_communication_available l2)
      (idle rover1)
      (navcam_ready rover1)
      (location pic_req1 l3)
      (pending pic_req1)
      (location pic_req2 l2)
      (location pic_req3 l5)
      (= (traversal_time unknown l1) 101)
      (= (traversal_time unknown l3) 3)
      (= (traversal_time unknown l2) 169)
      (= (traversal_time unknown l5) 281)
      (= (traversal_time unknown l4) 160)
      (= (traversal_time l1 l4) 122)
      (= (traversal_time l4 l2) 166)
      (= (traversal_time l5 l3) 284)
      (= (traversal_time l5 l1) 258)
      (= (traversal_time l1 l1) 0)
      (= (traversal_time l2 l1) 217)
      (= (traversal_time l1 l5) 258)
      (= (traversal_time l4 l1) 122)
      (= (traversal_time l3 l4) 163)
      (= (traversal_time l2 l5) 202)
      (= (traversal_time l4 l5) 137)
      (= (traversal_time l1 l2) 217)
      (= (traversal_time l3 l2) 171)
      (= (traversal_time l3 l1) 105)
      (= (traversal_time l3 l5) 284)
      (= (traversal_time l2 l4) 166)
      (= (traversal_time l4 l4) 0)
      (= (traversal_time l1 l3) 105)
      (= (traversal_time l5 l2) 202)
      (= (traversal_time l2 l3) 171)
      (= (traversal_time l5 l4) 137)
      (= (traversal_time l3 l3) 0)
      (= (traversal_time l4 l3) 163)
      (= (traversal_time l2 l2) 0)
      (= (traversal_time l5 l5) 0)
      (at 610 (not (orbiter_communication_available l2)))
      (at 610 (not (orbiter_communication_available l4)))
    )
    (:goal
      (and
        (transmitted pic_req1)
(transmitted pic_req2)
(transmitted rock_req1)
(transmitted pic_req3)
(transmitted rock_req2)
      )
    )
)
