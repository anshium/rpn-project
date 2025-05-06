import math
import pybullet as p
import time
import pybullet_data

def spawn_sphere_with_velocity(position, linear_velocity, radius=0.1, mass=0.1, color=[1, 0, 0, 1], physics_client_id=0):
    if mass < 0:
        print("Warning: Mass cannot be negative. Setting mass to 0 (static).")
        mass = 0
    
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=physics_client_id
    )
    
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        physicsClientId=physics_client_id
    )
    
    sphere_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position,
        baseOrientation=[0, 0, 0, 1],  
        physicsClientId=physics_client_id
    )
    
    if mass > 0:
        p.resetBaseVelocity(
            objectUniqueId=sphere_id,
            linearVelocity=linear_velocity,
            angularVelocity=[0, 0, 0],  
            physicsClientId=physics_client_id
        )
    elif mass == 0:
        p.resetBaseVelocity(sphere_id, [0,0,0], [0,0,0], physicsClientId=physics_client_id)

    return sphere_id

def predict_landing_point(initial_position, initial_velocity, gravity_z=-9.81, ground_z=0.0):

    x0, y0, z0 = initial_position
    vx0, vy0, vz0 = initial_velocity

    h0 = z0 - ground_z

    time_to_impact = None

    if abs(gravity_z) < 1e-6: # Effectively no gravity
        if abs(vz0) < 1e-6: # No vertical velocity
            if abs(h0) < 1e-6: # Already on the ground and not moving vertically
                time_to_impact = 0.0
            else: # Floating, will never hit ground_z
                return None, None, None, None
        elif vz0 < 0 and h0 >=0 : # Moving towards ground_z from above or at ground_z
            time_to_impact = -h0 / vz0
        elif vz0 > 0 and h0 <=0: # Moving towards ground_z from below or at ground_z
             time_to_impact = -h0 / vz0 # will be positive
        else: # Moving away from ground_z or parallel and not at ground_z
            return None, None, None, None
    else: # With gravity
        # Coefficients for the quadratic equation
        a_quad = 0.5 * gravity_z
        b_quad = vz0
        c_quad = h0

        discriminant = b_quad**2 - 4 * a_quad * c_quad

        if discriminant < 0:
            # No real solutions for t, means it won't reach ground_z (e.g., thrown upwards and never comes down to ground_z)
            # Or already below ground and moving further down without crossing ground_z from above.
            # This check is more nuanced if it starts below ground.
            # For simplicity, if it starts above ground and discriminant is negative, it means it won't come down TO ground_z.
            if h0 > 1e-6 : # If started above ground and won't reach it
                 return None, None, None, None
            # If started at or below ground and moving away, discriminant might also be <0
            # but we are interested in positive time solutions if it were to cross.
            # This case needs careful handling if the projectile can start below ground_z.
            # For now, assuming we usually care about impact from above or at ground_z.
            # A simple check for already being on the ground:
            if abs(h0) < 1e-6 and vz0 <= 0: # On ground and not moving up
                time_to_impact = 0.0
            else: # More complex scenario or truly won't hit
                return None, None, None, None


        if time_to_impact is None: # if not set by special conditions above
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b_quad + sqrt_discriminant) / (2 * a_quad)
            t2 = (-b_quad - sqrt_discriminant) / (2 * a_quad)

            # We need the positive, future time.
            # If gravity_z is negative (acting downwards), a_quad is negative.
            # t1 usually corresponds to an earlier time (e.g., if thrown from ground up),
            # t2 usually corresponds to the later impact time.
            
            # Filter for valid, positive times
            valid_times = []
            if t1 >= -1e-6: # Allow slightly negative from float precision if it should be 0
                valid_times.append(max(0, t1)) # clamp to 0 if very slightly negative
            if t2 >= -1e-6:
                valid_times.append(max(0, t2))

            if not valid_times:
                if abs(h0) < 1e-6 and vz0 <= 0 : # On ground and not moving up
                    time_to_impact = 0.0
                else:
                    return None, None, None, None # No valid future impact time
            else:
                # If it starts on the ground (h0 approx 0), one time will be ~0.
                # If vz0 > 0 (thrown up from ground), we want the other positive time.
                # If vz0 <= 0 (thrown down/horizontally from ground), t=0 is the impact.
                if abs(h0) < 1e-6 and vz0 <= 0:
                    time_to_impact = 0.0
                else:
                    # Select the smallest positive time if multiple, or the only positive one.
                    # If thrown upwards from above ground, there will be two positive roots if the
                    # trajectory equation were solved for z0 again, but we solve for ground_z.
                    # We usually want the larger positive root when thrown up and landing.
                    # However, if the object starts below ground_z and is thrown upwards,
                    # it might cross ground_z twice. We want the first positive time it hits ground_z.
                    # Let's take the smallest non-negative time if there are multiple.
                    # But for typical projectile (start above or at ground, land on ground), if two positive, it's usually the larger one.
                    # Consider t1 and t2:
                    # If gravity_z is negative, a_quad is negative.
                    # (-b + sqrt(D)) / (2a)  vs  (-b - sqrt(D)) / (2a)
                    # The one with -sqrt(D) in numerator will be "more negative" / "less positive".
                    # The one with +sqrt(D) in numerator will be "less negative" / "more positive".
                    # So if 2a is negative, t1 is "less positive" or "more negative", t2 is "more positive" or "less negative"
                    # For impact time, we generally want the larger positive root if thrown upwards.
                    # Or the only positive root if thrown downwards.
                    positive_times = [t for t in [t1, t2] if t >= -1e-6] # use max(0,t) later
                    if not positive_times:
                         return None, None, None, None # Should be caught by discriminant or earlier checks

                    # If starting at ground and thrown up, one t is ~0, other is flight time.
                    # If starting above ground, there should be one positive impact time.
                    time_to_impact = max(t for t in positive_times if t is not None) # Get the largest positive time
                    if time_to_impact < 1e-6 and (h0 > 1e-6 or (abs(h0) < 1e-6 and vz0 > 0)):
                        # This handles case where we pick t=0 but it's thrown upwards from ground,
                        # or started above ground and max returned a t=0 which is wrong.
                        # Try to find a strictly positive time if available and reasonable
                        strictly_positive_times = [t for t in positive_times if t > 1e-6]
                        if strictly_positive_times:
                            time_to_impact = min(strictly_positive_times) # If thrown up from below ground, first hit
                                                                         # Or if thrown up from ground, the non-zero time
                            # If thrown from above ground, it should be the only positive one.
                            # Let's simplify: take the LARGEST non-negative solution.
                            # This is generally correct for a projectile launched from z0 >= ground_z.
                            time_to_impact = -1 # reset
                            for t_sol in [t1, t2]:
                                if t_sol >= -1e-6: # consider it valid if non-negative
                                    if time_to_impact < 0 or t_sol > time_to_impact : # find largest non-negative
                                        time_to_impact = max(0, t_sol)


    if time_to_impact is None or time_to_impact < 0: # Check again if any path failed to set it
        if abs(h0) < 1e-6 and vz0 <= 0: time_to_impact = 0.0 # final check for on ground
        else: return None, None, None, None


    # --- Calculate landing position (x_impact, y_impact) ---
    landing_x = x0 + vx0 * time_to_impact
    landing_y = y0 + vy0 * time_to_impact

    return landing_x, landing_y, ground_z, time_to_impact


# --- Example Usage ---
if __name__ == "__main__":
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    g_z = -9.81
    p.setGravity(0, 0, g_z, physicsClientId=client_id)
    p.setRealTimeSimulation(0, physicsClientId=client_id)
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)

    # Test case 1: Dropped from rest
    initial_pos1 = [0, 0, 2]
    initial_vel1 = [0, 0, 0]
    lx1, ly1, lz1, t1 = predict_landing_point(initial_pos1, initial_vel1, gravity_z=g_z)
    if lx1 is not None:
        print(f"Test 1 (Drop): Landing at ({lx1:.2f}, {ly1:.2f}, {lz1:.2f}) at t={t1:.2f}s")
        spawn_sphere_with_velocity(initial_pos1, initial_vel1, radius=0.05, mass=0.1, color=[1,0,0,1], physics_client_id=client_id)
        # Mark predicted spot
        p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1,0,0,0.5], visualFramePosition=[lx1, ly1, lz1+0.01])
    else:
        print("Test 1 (Drop): Will not land.")


    # Test case 2: Thrown horizontally
    initial_pos2 = [-1, 0.5, 1.5]
    initial_vel2 = [1.0, 0.5, 0]
    lx2, ly2, lz2, t2 = predict_landing_point(initial_pos2, initial_vel2, gravity_z=g_z)
    if lx2 is not None:
        print(f"Test 2 (Horizontal): Landing at ({lx2:.2f}, {ly2:.2f}, {lz2:.2f}) at t={t2:.2f}s")
        spawn_sphere_with_velocity(initial_pos2, initial_vel2, radius=0.05, mass=0.1, color=[0,1,0,1], physics_client_id=client_id)
        p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0,1,0,0.5], visualFramePosition=[lx2, ly2, lz2+0.01])
    else:
        print("Test 2 (Horizontal): Will not land.")

    # Test case 3: Thrown upwards and forwards
    initial_pos3 = [0.5, -0.8, 1.0]
    initial_vel3 = [0.5, 0.3, 3.0] # Positive vz0
    lx3, ly3, lz3, t3 = predict_landing_point(initial_pos3, initial_vel3, gravity_z=g_z)
    if lx3 is not None:
        print(f"Test 3 (Upwards): Landing at ({lx3:.2f}, {ly3:.2f}, {lz3:.2f}) at t={t3:.2f}s")
        spawn_sphere_with_velocity(initial_pos3, initial_vel3, radius=0.05, mass=0.1, color=[0,0,1,1], physics_client_id=client_id)
        p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0,0,1,0.5], visualFramePosition=[lx3, ly3, lz3+0.01])
    else:
        print("Test 3 (Upwards): Will not land.")

    # Test case 4: Starting on ground, thrown horizontally (should be t=0)
    initial_pos4 = [-1, -1, 0.001] # Slightly above to avoid initial penetration issues in prediction
    initial_vel4 = [1, 0, 0]
    lx4, ly4, lz4, t4 = predict_landing_point(initial_pos4, initial_vel4, gravity_z=g_z, ground_z=0.0)
    if lx4 is not None:
        print(f"Test 4 (On ground, horizontal): Landing at ({lx4:.2f}, {ly4:.2f}, {lz4:.2f}) at t={t4:.2f}s")
        spawn_sphere_with_velocity(initial_pos4, initial_vel4, radius=0.05, mass=0.1, color=[1,1,0,1], physics_client_id=client_id)
        p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1,1,0,0.5], visualFramePosition=[lx4, ly4, lz4+0.01])
    else:
        print("Test 4 (On ground, horizontal): Will not land.")


    p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0.5], physicsClientId=client_id)

    try:
        for i in range(1000): # Simulate for a while
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1./240.)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect(physicsClientId=client_id)