(define (domain RSS_Project_Rover_Scenario)
    (:requirements :typing)
    (:types
      Rover - object
      Request - object
      RockSampleRequest - Request
      PictureRequest - Request
      Location - object
    )
    (:predicates
      (mastcam_ready ?rov - Rover)
      (hazcam_ready ?rov - Rover)
      (idle ?rov - Rover)
      (mastcam_on ?rov - Rover)
      (hazcam_on ?rov - Rover)
      (pending ?req - Request)
      (done ?req - Request)
      (transmitted ?req - Request)
      (has_rock ?loc - Location)
      (orbiter_communication_available ?loc - Location)
      (surveyed ?loc - Location)
      (at ?rov - Rover ?loc - Location)
      (location ?pic - PictureRequest ?loc - Location)

      (mastcam_issue ?rov - Rover)

      ;; Test: avoid the use of timed initial literals for transmitting requested data
      (communication_available ?loc - Location)
    )

    (:action move
     :parameters (?self - Rover ?loc1 - Location ?loc2 - Location)
     :precondition
       (and
         (at ?self ?loc1)
         (idle ?self)
       )
     :effect
       (and
         (not (at ?self ?loc1))
         (at ?self ?loc2)
         (idle ?self)
       )
    )

    (:action turnon_mastcam
     :parameters (?self - Rover ?loc - Location)
     :precondition
       (and
         (at ?self ?loc)
         (idle ?self)
         (mastcam_ready ?self)
       )
     :effect
       (and
         (not (idle ?self))
         (mastcam_on ?self)
       )
    )

    (:action turnon_hazcam
     :parameters (?self - Rover ?loc - Location)
     :precondition
       (and
         (at ?self ?loc)
         (idle ?self)
         (hazcam_ready ?self)
         (mastcam_issue ?self)
       )
     :effect
       (and
         (not (idle ?self))
         (hazcam_on ?self)
       )
    )

    (:action take_pictures_mastcam
     :parameters (?self - Rover ?loc - Location ?req - PictureRequest)
     :precondition
       (and
         (at ?self ?loc)
         (mastcam_on ?self)
         (location ?req ?loc)
         (pending ?req)
       )
     :effect
       (and
         (done ?req)
         (idle ?self)
         (not (mastcam_on ?self))
         (idle ?self)
       )
    )

    (:action take_pictures_hazcam
     :parameters (?self - Rover ?loc - Location ?req - PictureRequest)
     :precondition
       (and
         (at ?self ?loc)
         (hazcam_on ?self)
         (location ?req ?loc)
         (pending ?req)
         (mastcam_issue ?self)
       )
     :effect
       (and
         (not (pending ?req))
         (done ?req)
         (not (hazcam_on ?self))
         (idle ?self)
       )
    )

    (:action survey_location
     :parameters (?self - Rover ?loc - Location)
     :precondition
       (and
         (at ?self ?loc)
         (idle ?self)
       )
     :effect
       (and
         (surveyed ?loc)
         (not (idle ?self))
       )
    )

    (:action collect_rock_sample
     :parameters (?self - Rover ?loc - Location ?req - RockSampleRequest)
     :precondition
       (and
         (at ?self ?loc)
         (pending ?req)
         (has_rock ?loc)
         (surveyed ?loc)
       )
     :effect
       (and
         (not (pending ?req))
         (done ?req)
         (not (has_rock ?loc))
         (idle ?self)
       )
    )

    (:action transmit_data
     :parameters (?self - Rover ?loc - Location ?req - Request)
     :precondition
       (and
         (at ?self ?loc)
         (done ?req)
         (orbiter_communication_available ?loc)
	       ;;(communication_available ?loc)
         (idle ?self)
       )
     :effect
         (and
            (transmitted ?req)
         )
    )


)

