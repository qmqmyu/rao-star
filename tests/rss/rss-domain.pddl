(define (domain RSS_Project_Rover_Scenario)
    (:requirements :typing :durative-actions :fluents :timed-initial-literals)
    (:types
      Rover - object
      Request - object
      RockSampleRequest - Request
      PictureRequest - Request
      Location - object
    )
    (:predicates
      (mastcam_ready ?rov - Rover)
      (navcam_ready ?rov - Rover)
      (idle ?rov - Rover)
      (mastcam_on ?rov - Rover)
      (navcam_on ?rov - Rover)
      (pending ?req - Request)
      (done ?req - Request)
      (transmitted ?req - Request)
      (has_rock ?loc - Location)
      (orbiter_communication_available ?loc - Location)
      (surveyed ?loc - Location)
      (at ?rov - Rover ?loc - Location)
      (location ?pic - PictureRequest ?loc - Location)
    )
    (:functions
      (traversal_time ?from - Location ?to - Location)
    )

    (:durative-action take_pictures_mastcam
     :parameters (?self - Rover ?loc - Location ?req - PictureRequest)
     :duration (= ?duration 20)
     :condition
       (and
         (at start (at ?self ?loc))
         (over all (at ?self ?loc))
         (at start (mastcam_on ?self))
         (at start (location ?req ?loc))
         (at start (pending ?req))
       )
     :effect
       (and
         (at start (not (pending ?req)))
         (at end (done ?req))
         (at start (not (idle ?self)))
         (at end (idle ?self))
         (at end (not (mastcam_on ?self)))
       )
    )

    (:durative-action move
     :parameters (?self - Rover ?loc1 - Location ?loc2 - Location)
     :duration (= ?duration (traversal_time ?loc1 ?loc2))
     :condition
       (and
         (at start (at ?self ?loc1))
         (at start (idle ?self))
       )
     :effect
       (and
         (at start (not (at ?self ?loc1)))
         (at end (at ?self ?loc2))
       )
    )

    (:durative-action take_pictures_navcam
     :parameters (?self - Rover ?loc - Location ?req - PictureRequest)
     :duration (= ?duration 20)
     :condition
       (and
         (at start (at ?self ?loc))
         (over all (at ?self ?loc))
         (at start (navcam_on ?self))
         (at start (location ?req ?loc))
         (at start (pending ?req))
       )
     :effect
       (and
         (at end (not (pending ?req)))
         (at end (done ?req))
         (at start (not (idle ?self)))
         (at end (idle ?self))
         (at end (not (navcam_on ?self)))
       )
    )

    (:durative-action transmit_data
     :parameters (?self - Rover ?loc - Location ?req - Request)
     :duration (= ?duration 30)
     :condition
       (and
         (at start (at ?self ?loc))
         (over all (at ?self ?loc))
         (at start (done ?req))
         (at start (orbiter_communication_available ?loc))
         (over all (orbiter_communication_available ?loc))
         (at start (idle ?self))
       )
     :effect
       (and
         (at end (transmitted ?req))
         (at start (not (idle ?self)))
         (at end (idle ?self))
       )
    )

    (:durative-action collect_rock_sample
     :parameters (?self - Rover ?loc - Location ?req - RockSampleRequest)
     :duration (= ?duration 50)
     :condition
       (and
         (at start (at ?self ?loc))
         (over all (at ?self ?loc))
         (at start (pending ?req))
         (at start (has_rock ?loc))
         (at start (surveyed ?loc))
         ;;(at start (idle ?self))
       )
     :effect
       (and
         (at end (not (pending ?req)))
         (at end (done ?req))
         (at start (not (has_rock ?loc)))
         ;;(at start (not (idle ?self)))
         (at end (idle ?self))
       )
    )

    (:durative-action survey_location
     :parameters (?self - Rover ?loc - Location)
     :duration (= ?duration 50)
     :condition
       (and
         (at start (at ?self ?loc))
         (over all (at ?self ?loc))
         (at start (idle ?self))
       )
     :effect
       (and
         (at end (surveyed ?loc))
         (at start (not (idle ?self)))
         ;;(at end (idle ?self)) ;;commented this because popf was adding this action for no reason
       )
    )

    (:durative-action turnon_mastcam
     :parameters (?self - Rover ?loc - Location)
     :duration (= ?duration 20)
     :condition
       (and
         (at start (at ?self ?loc))
         (over all (at ?self ?loc))
         (at start (idle ?self))
         (at start (mastcam_ready ?self))
       )
     :effect
       (and
         (at start (not (idle ?self)))
         (at end (mastcam_on ?self))
       )
    )

    (:durative-action turnon_navcam
     :parameters (?self - Rover ?loc - Location)
     :duration (= ?duration 20)
     :condition
       (and
         (at start (at ?self ?loc))
         (over all (at ?self ?loc))
         (at start (idle ?self))
         (at start (navcam_ready ?self))
       )
     :effect
       (and
         (at start (not (idle ?self)))
         (at end (navcam_on ?self))
       )
    )

    ;;#(:durative-action start_event_orbiter_communication_available
    ;;# :parameters (?p - Location)
    ;;# :duration (= ?duration 0.005)
    ;;# :effect
    ;;#     (and
    ;;#       (at start (orbiter_communication_available ?p))))

    ;;#(:durative-action end_event_orbiter_communication_available
    ;;# :parameters (?p - Location)
    ;;# :duration (= ?duration 0.005)
    ;;# :effect
    ;;#     (and
    ;;#       (at start (not (orbiter_communication_available ?p)))))

)
