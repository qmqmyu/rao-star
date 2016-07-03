(define (domain mitsubishi-domain)
  (:requirements :strips :typing)
    (:types
        manipulator - object
        thing - object
        location - object)
    (:predicates
                    (empty ?manip - manipulator)
                    (clear ?obj - thing)
                    (free ?loc - location)                    
                    (at ?obj - thing ?loc - location)
                    (holding ?obj - thing ?manip - manipulator)
                    (reachable ?loc - location ?manip - manipulator)
                    (clean ?loc - location)
                    (canSolder ?obj - thing ?loc - location)
                    (soldered ?obj - thing)
                    (isSolder ?obj - thing)
                    (isCleaner ?obj - thing)
                    (isHuman ?manip - manipulator)
                    (isComponent ?obj - thing))

    (:action pick
      :parameters (?obj - thing ?manip - manipulator ?loc - location)
      :precondition (and (clear ?obj) (at ?obj ?loc) (empty ?manip) (reachable ?loc ?manip))
      :effect (and (holding ?obj ?manip) (free ?loc) (not (clear ?obj)) (not (at ?obj ?loc)) (not (empty ?manip)))
      )

    (:action place
      :parameters  (?obj - thing ?manip - manipulator ?loc - location)
      :precondition (and (holding ?obj ?manip) (free ?loc))
      :effect (and (clear ?obj) (empty ?manip) (at ?obj ?loc) (not (holding ?obj ?manip)) (not (free ?loc)))
      )

    (:action clean
      :parameters  (?loc - location ?manip - manipulator ?obj - thing)
      :precondition (and (holding ?obj ?manip) (free ?loc) (isCleaner ?obj))
      :effect (clean ?loc)
      )

    (:action solder
      :parameters  (?obj - thing ?manip - manipulator ?solderinglocation - location ?sold - thing)
      :precondition (and (at ?obj ?solderinglocation) (canSolder ?obj ?solderinglocation) (clear ?obj) (holding ?sold ?manip) (clean ?solderinglocation) (isComponent ?obj) (isSolder ?sold) (isHuman ?manip))
      :effect (soldered ?obj)
      )

    (:action pass
      :parameters (?obj - thing ?manip1 - manipulator ?manip2 - manipulator)
      :precondition (and (holding ?obj ?manip1) (empty ?manip2) (isHuman ?manip2))
      :effect (and (holding ?thing ?manip2) (not (holding ?thing ?manip1)) (not (empty ?manip2)) (empty ?manip1))
      ))
