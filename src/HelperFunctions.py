import os
import json
import math

class Utilities:
    Tolerance = 0.0001
    DecimalPoints = 2

    @staticmethod
    def LoadJSONfiles(root_folder, label):
        result = {}

        # Walk through all subfolders
        for dirpath, _, filenames in os.walk(root_folder):
            subfolder_name = os.path.basename(dirpath)

            for fname in filenames:
                if fname.lower().endswith(".json"):
                    json_path = os.path.join(dirpath, fname)

                    # Key: subfoldername_endfix
                    endfix = os.path.splitext(fname)[0]
                    key = f"{label}_{subfolder_name}_{endfix}"

                    # Load json file
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    result[key] = data

        # Sort dictionary by key (ascending)
        sorted_result = dict(sorted(result.items(), key=lambda x: x[0]))
        return sorted_result

class Vector():
    
    @staticmethod
    def VecCrossProd(u, v):
        return (
            u[1]*v[2] - u[2]*v[1],
            u[2]*v[0] - u[0]*v[2],
            u[0]*v[1] - u[1]*v[0],
        )

    @staticmethod
    def VecDotProd(u, v):
        return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

    @staticmethod
    def VecSubtract(u, v):
        return (u[0]-v[0], u[1]-v[1], u[2]-v[2])

    @staticmethod
    def VecAdd(u, v):
        return (u[0]+v[0], u[1]+v[1], u[2]+v[2])

    @staticmethod
    def VecScale(k, u):
        return (u[0]*k, u[1]*k, u[2]*k)

    @staticmethod
    def VecNorm(u):
        return (u[0]**2 + u[1]**2 + u[2]**2) ** 0.5
    
    @staticmethod
    def VecAngleBetween(u, v):
        # Dot product
        dot = sum(ui * vi for ui, vi in zip(u, v))
        
        # Norms
        norm_u = math.sqrt(sum(ui**2 for ui in u))
        norm_v = math.sqrt(sum(vi**2 for vi in v))
        
        # Avoid division by zero
        if norm_u == 0 or norm_v == 0:
            raise ValueError("Zero-length vector provided.")
        
        # Cosine of angle
        cosang = dot / (norm_u * norm_v)
        
        # Clamp for numerical safety
        cosang = max(min(cosang, 1.0), -1.0)
        
        # Angle in degrees
        return math.degrees(math.acos(cosang))

class Geometry():
    
    @staticmethod
    def EuclideanDistance(vector1, vector2):
        if len(vector1) != len(vector2):
            raise ValueError("Points must have the same dimension.")
        
        squared_diffs = [(a - b) ** 2 for a, b in zip(vector1, vector2)]
        return math.sqrt(sum(squared_diffs))
    
    #################################
    ### I N T E R S E C T I O N S ###
    #################################

    @staticmethod
    def SegSegIntersection(p1, p2, q1, q2, strict=True):
        d1 = Vector.VecSubtract(p2, p1)
        d2 = Vector.VecSubtract(q2, q1)
        r  = Vector.VecSubtract(q1, p1)

        cross_d1d2 = Vector.VecCrossProd(d1, d2)
        norm_cross = Vector.VecNorm(cross_d1d2)

        if norm_cross < Utilities.Tolerance:
            # Parallel or coincident
            cross_r_d1 = Vector.VecCrossProd(r, d1)
            if Vector.VecNorm(cross_r_d1) < Utilities.Tolerance:
                # coincident
                return False, None
            else:
                # parallel
                return False, None

        # Solve for t, s using cross products (avoids linear algebra library)
        # From formula: t = ( (q1 - p1) × d2 ) ⋅ (d1 × d2) / |d1 × d2|^2
        cross_rd2 = Vector.VecCrossProd(r, d2)
        t = Vector.VecDotProd(cross_rd2, cross_d1d2) / (norm_cross**2)

        cross_rd1 = Vector.VecCrossProd(r, d1)
        s = Vector.VecDotProd(cross_rd1, cross_d1d2) / (norm_cross**2)

        if strict==True:
            # Restrict to finite segments: t, s must be in [0,1]
            if not (0 - Utilities.Tolerance <= t <= 1 + Utilities.Tolerance and 0 - Utilities.Tolerance <= s <= 1 + Utilities.Tolerance):
                return False, None

        p_int1 = Vector.VecAdd(p1, Vector.VecScale(t, d1))
        p_int2 = Vector.VecAdd(q1, Vector.VecScale(s, d2))

        if Vector.VecNorm(Vector.VecSubtract(p_int1, p_int2)) < Utilities.Tolerance:
            return True, p_int1
        else:
            return False, None
    
    @staticmethod
    def LineLineIntersection(p1, p2, q1, q2):
        return Geometry.SegSegIntersection(p1,p2,q1,q2,False)
