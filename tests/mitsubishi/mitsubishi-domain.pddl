(define (domain mitsubishi-domain)
  (:requirements :strips :typing :durative-actions)
    (:types manipulator object location)
    (:predicates                        
                    (empty ?manip - manipulator)
                    (clear ?obj - object)
                    (at ?obj - object ?loc - location)
                    (holding ?obj - object ?manip - manipulator)
                    (reachable ?loc - location ?manip - manipulator)                    
                    (clean ?loc - location)
                    (canSolder ?obj - object ?loc - location)
                    (soldered ?obj - object)
                    (isSolder ?obj - object)
                    (isCleaner ?obj - object)
                    (isHuman ?manip - manipulator)
                    (isComponent ?obj - object))

    (:durative-action pick
      :parameters (?obj - object ?manip - manipulator ?loc - location)
      :precondition (and (clear ?obj) (at ?obj ?loc) (empty ?manip) (reachable ?loc ?manip))
      :effect (and (holding ?obj ?manip) (not (clear ?obj)) (not (at ?obj ?loc)) (not (empty ?manip)))
      :duration (u[1,10]))
     
    (:durative-action place
      :parameters  (?obj - object ?manip - manipulator ?loc - location)
      :precondition (holding ?obj ?manip)
      :effect (and (clear ?obj) (empty ?manip) (at ?obj ?loc) (not (holding ?obj ?manip)))
      :duration (u[3,15]))

    (:durative-action clean
      :parameters  (?loc - location ?manip - manipulator ?obj - object)
      :precondition (and (holding ?obj ?manip) (isCleaner ?obj))
      :effect (clean ?loc)
      :duration ([5,10]))

    (:durative-action solder
      :parameters  (?obj - object ?manip - manipulator ?solderinglocation - location ?sold - object)
      :precondition (and (at ?obj ?solderinglocation) (canSolder ?obj ?solderinglocation) (clear ?obj) (holding ?sold ?manip) (clean ?solderinglocation) (isComponent ?obj) (isSolder ?sold) (isHuman ?manip))
      :effect (soldered ?obj)
      :duration (u[10,20]))
                   
    (:durative-action pass
      :parameters (?obj - object ?manip1 - manipulator ?manip2 - manipulator)
      :precondition (and (holding ?obj ?manip1) (empty ?manip2) (isHuman ?manip2))
      :effect (and (holding ?object ?manip2) (not (holding ?object ?manip1)) (not (empty ?manip2)) (empty ?manip1))
      :duration (u[10,20])))
