
def sphere(r, points = 100):
    import numpy as np
    u = np.linspace(0, 2 * np.pi, points)
    v = np.linspace(0, np.pi, points)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x,y,z

def lattice(latt_a, latt_b, latt_c, scale = 1):
    import numpy as np
    latt_a, latt_b, latt_c = latt_a *scale , latt_b * scale, latt_c * scale
    points = []
    o = np.array((0,0,0))
    oa = o + latt_a
    ob = o + latt_b
    oc = o + latt_c
    oab = oa + latt_b
    oac = oa + latt_c
    obc = ob + latt_c
    oabc = oab + latt_c
    middle = 0.5* oabc
    o, oa, ob, oc, oab, oac, obc, oabc = o - middle, oa- middle, ob- middle, oc- middle, oab- middle, oac- middle, obc- middle, oabc- middle
    x = np.array([[o[0],oc[0]],[oa[0],oac[0]],[oab[0],oabc[0]],[ob[0],obc[0]],[o[0],oc[0]],[o[0],oa[0]],[ob[0],oab[0]],[obc[0],oabc[0]],[oc[0],oac[0]],[o[0],oa[0]]])
    y = np.array([[o[1],oc[1]],[oa[1],oac[1]],[oab[1],oabc[1]],[ob[1],obc[1]],[o[1],oc[1]],[o[1],oa[1]],[ob[1],oab[1]],[obc[1],oabc[1]],[oc[1],oac[1]],[o[1],oa[1]]])
    z = np.array([[o[2],oc[2]],[oa[2],oac[2]],[oab[2],oabc[2]],[ob[2],obc[2]],[o[2],oc[2]],[o[2],oa[2]],[ob[2],oab[2]],[obc[2],oabc[2]],[oc[2],oac[2]],[o[2],oa[2]]])
    
    return x,y,z

def plane_vector(hkl, latt_a, latt_b, latt_c, r= 1):
    import numpy as np
    h,k,l = hkl
    x, y, z, u, v, w = [], [], [], [], [], []
    mid_point = (latt_a + latt_b + latt_c)/2
    #shifting the origin allows us to get the correct points for negative indices
    shifted_origin = np.sign(h)*(latt_a*-0.5*(np.sign(h)-1)) + np.sign(k)*(latt_b*-0.5*(np.sign(k)-1)) + np.sign(l)*(latt_c*-0.5*(np.sign(l)-1)) 
    if h != 0:
        h_point = latt_a/ h - shifted_origin
        if k != 0:
            k_point = latt_b/ k - shifted_origin
            if l != 0:
                l_point = latt_c/ l - shifted_origin
                hk = h_point - k_point
                hl = h_point - l_point
                normal = np.cross(hl, hk)  *np.sign(h)*np.sign(k)*np.sign(l)
                normal = 1/np.linalg.norm(normal) * normal
                point = h_point
            else:
                hk = h_point - k_point
                normal = np.cross(hk, latt_c) *np.sign(h)*np.sign(k)
                normal = 1/np.linalg.norm(normal) * normal
                point = h_point
        else:
            if l != 0:
                l_point = latt_c/ l - shifted_origin
                hl = h_point - l_point
                normal = np.cross(latt_b, hl) *np.sign(h)*np.sign(l)
                normal = 1/np.linalg.norm(normal) * normal
                point = h_point
            else:
                normal = np.cross(latt_b, latt_c)* np.sign(h)
                normal = 1/np.linalg.norm(normal) * normal
                point = h_point
    else:
        if k != 0:
            k_point = latt_b/ k - shifted_origin
            if l != 0:
                l_point = latt_c/ l - shifted_origin
                kl = k_point - l_point
                normal = np.cross(kl, latt_a)*np.sign(l)*np.sign(k)
                normal = 1/np.linalg.norm(normal) * normal
                point = k_point
            else:
                normal = np.cross(latt_c, latt_a)* np.sign(k)
                normal = 1/np.linalg.norm(normal) * normal
                point = k_point
        else:
            if l != 0:
                l_point = latt_c/ l - shifted_origin
                normal = np.cross(latt_a, latt_b)* np.sign(l)
                normal = 1/np.linalg.norm(normal) * normal
                point = l_point
            else:
                normal = np.array((0,0,1))
                point = mid_point
    x.append(point[0]), y.append(point[1]), z.append(point[2]), u.append(normal[0]), v.append(normal[1]), w.append(normal[2])
    return x,y,z,u,v,w

def plane(hkl, latt_a, latt_b, latt_c, r=1, scale = 1):
    import numpy as np
    h, k, l = hkl
    x,y,z,u,v,w = plane_vector(hkl, latt_a, latt_b, latt_c)
    mid_point = (latt_a + latt_b + latt_c)/2
    point  = np.array((x[0],y[0],z[0])) - mid_point
    normal = np.array((u[0],v[0],w[0]))
    mesh = np.meshgrid(np.linspace(-2*scale,2*scale,10), np.linspace(-2*scale,2*scale,10))
    
    if l !=0:
        # a plane is a*x+b*y+c*z+d=0
        # [a,b,c] is the normal. Thus, we have to calculate
        # d and we're set
        d = -point.dot(normal)

        # create x,y
        xx, yy = mesh

        # calculate corresponding z
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
        
    elif h !=0:
        d = -point.dot(normal)

        # create x,y
        yy, zz = mesh

        # calculate corresponding z
        xx = (-normal[2] * zz - normal[1] * yy - d) * 1. /normal[0]
    
    elif k !=0:
        d = -point.dot(normal)

        # create x,y
        xx, zz = mesh

        # calculate corresponding z
        yy = (-normal[2] * zz - normal[0] * xx - d) * 1. /normal[1]
        
    if h == k == l == 0:
        point += mid_point
        xx, yy = mesh
        
        zz = np.zeros_like(xx)
    
    return xx,yy,zz

def equatorial_projection(vector):
    import numpy as np
    x,y,z,u,v,w = vector
    u,v,w = u[0],v[0],w[0]
    if w >= 0 :
        pole = (0,0,-1)
    if w < 0:
        pole = (0,0,1)
    r = (-w)/(w-pole[2])
    nx,ny = u + r*u, v + r*v
    line = ((u,v,w), pole)
    return nx,ny, line
    
def plane_sphere_intercept(hkl, latt_a, latt_b, latt_c):
    import numpy as np
    h,k,l = hkl
    x,y,z,u,v,w = plane_vector(hkl, latt_a, latt_b, latt_c)
    mid_point = (latt_a + latt_b + latt_c)/2
    point  = np.array((x[0],y[0],z[0])) - mid_point
    normal = np.array((u[0],v[0],w[0]))
    len_o_to_plane = point.dot(normal)
    shift_vector = len_o_to_plane * normal
    intercept = np.array((0, np.sqrt(1 - len_o_to_plane**2), -len_o_to_plane))
    op = np.array((0,0, -len_o_to_plane))
    r = np.linalg.norm(intercept-op)
    
    
    
    
    phi = np.linspace(0, 2 * np.pi, 100)
    i = r *np.cos(phi)
    j = r* np.sin(phi)
    m = np.zeros(np.size(phi))
    xx, yy, zz = [],[],[]
    
    new_vec = 1

    rot_axis = np.array((0.5*normal[0], 0.5*normal[1], 0.5*normal[2]+0.5))
    if np.linalg.norm(rot_axis) == 0:
        rot_axis = np.array((0,0,1))
    normalise = 1/np.linalg.norm(rot_axis)
    rot_axis = normalise*rot_axis
    u,v,w = rot_axis[0], rot_axis[1], rot_axis[2]
    for each in range(len(i)):
        x,y,z = i[each], j[each], m[each]
        xx.append((2*u*(u*x + v*y + w*z)-x) + shift_vector[0]), yy.append((2*v*(u*x + v*y + w*z)- y)+ shift_vector[1]), zz.append((2*w*(u*x + v*y + w*z)- z)+ shift_vector[2])

    return xx,yy,zz

def shortest_path(p1, p2, hemisphere_correct = False):
    import numpy as np
    #first def the plane they lie in
    
    if hemisphere_correct == True:
        if p1[2]*p2[2] != 0:
            p2 *= -1
    
    normal = np.cross(p1, p2)
    if np.linalg.norm(normal) == 0:
        normal = p1
    normal = 1/np.linalg.norm(normal)* normal
    
    mp = (p1 + p2)/2
    d = np.linalg.norm(p1 - mp)
    
    r = 1

    phi = np.linspace(0, 2 * np.pi, 100)
    i = r *np.cos(phi)
    j = r* np.sin(phi)
    m = np.zeros(np.size(phi))
    xx, yy, zz = [],[],[]
   

    rot_axis = np.array((0.5*normal[0], 0.5*normal[1], 0.5*normal[2]+0.5))
    if np.linalg.norm(rot_axis) == 0:
        rot_axis = np.array((0,0,1))
    normalise = 1/np.linalg.norm(rot_axis)
    rot_axis = normalise*rot_axis
    u,v,w = rot_axis[0], rot_axis[1], rot_axis[2]
    for each in range(len(i)):
        x,y,z = i[each], j[each], m[each]
        nx, ny, nz = (2*u*(u*x + v*y + w*z)-x), (2*v*(u*x + v*y + w*z)- y),(2*w*(u*x + v*y + w*z)- z)
        npoint = np.array((nx, ny, nz))
        trial_d = np.linalg.norm(npoint - mp)
        if trial_d < d:
            xx.append(nx), yy.append(ny), zz.append(nz)
               
        
        

    return xx,yy,zz

def equatorial_trace_projection(points):
    import numpy as np
    xu, yu = [],[]
    i,j,k = points
    for each in range(len(points[0])):
        x,y,z = i[each], j[each], k[each]
    
        if z > 0 :
            pole = (0,0,-1)
            r = (-z)/(z-pole[2])
            xu.append(x + r*x), yu.append(y + r*y)
        if z < 0:
            pole = (0,0,1)
            r = (-z)/(z-pole[2])
            xu.append(x + r*x), yu.append(y + r*y)
        
    return xu,yu

def single_vector_display(lattice_vectors, hkl, factor=1e-6):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=3,wspace=0.7)
    ax = fig.add_subplot(gs[:, :-1], projection='3d')
    ax2 = fig.add_subplot(gs[0, -1])
    ax3 = fig.add_subplot(gs[1, -1])


    latt_a, latt_b, latt_c = lattice_vectors
    latt_a, latt_b, latt_c =latt_a*factor, latt_b*factor, latt_c*factor

    x,y,z = lattice(latt_a, latt_b, latt_c, 1)
    # Plot the surface
    ax.plot_surface(x,y,z, color='grey', alpha = 1)

    x,y,z = sphere(1,50)
    # Plot the surface
    ax.plot_surface(x,y,z, color='grey', alpha = 0.05)

    x,y,z,u,v,w = plane_vector(hkl, latt_a,latt_b,latt_c)
    ax.quiver(0,0,0,u,v,w, color = 'red')
    ax3.scatter(u[0],v[0], marker = 'x', color = 'red')

    projx,projy, line= equatorial_projection(plane_vector(hkl, latt_a,latt_b,latt_c))
    ax.scatter(projx,projy,0, marker = 'o', color = 'black')
    ax2.scatter(projx,projy, marker = 'o', color = 'black')
    
    head,tail = line
    ax.plot((head[0], tail[0]),(head[1], tail[1]),(head[2], tail[2]), ls = '--', color = 'grey')

    x,y,z = plane(hkl, latt_a, latt_b, latt_c)
    ax.plot_surface(x,y,z, color='blue', alpha = 0.3)

    x,y,z = plane_sphere_intercept(hkl, latt_a, latt_b, latt_c)
    ax.plot(x,y,z, color='black')
    ax3.plot(x,y, color='black', ls = '--')

    xu,yu= equatorial_trace_projection(plane_sphere_intercept(hkl, latt_a, latt_b, latt_c))
    ax2.plot(xu,yu, color='black')


    x,y,z = plane((0,0,0), latt_a, latt_b, latt_c)
    ax.plot_surface(x,y,z, color='red', alpha = 0.3)

    x,y,z = plane_sphere_intercept((0,0,0), latt_a, latt_b, latt_c)
    ax.plot(x,y,z, color='black', lw = 1)
    ax2.plot(x,y, color='black', lw = 1)
    ax3.plot(x,y, color='black', lw = 1)

    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax2.title.set_text('Stereographic Projection')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim([-1.2,1.2])
    ax2.set_ylim([-1.9,1.9])
    ax3.title.set_text('Orthographic Projection')
    ax3.set_xlim([-1.2,1.2])
    ax3.set_ylim([-1.9,1.9])
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.show()

    
def display_stereograph(lattice_vectors, hs, ks, ls, factor = 1e-6, omit = [], show_label = True, show_trace = True, four_index_notation = False, max_index = False):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=2,wspace=0.2)
    ax2 = fig.add_subplot(gs[:, :])


    latt_a, latt_b, latt_c = lattice_vectors
    latt_a, latt_b, latt_c =latt_a*factor, latt_b*factor, latt_c*factor
    
    label_dict = {}


    for h in hs:
        for k in ks:
            for l in ls:
                if h == k == l == 0:
                    continue
                else:
                    omit_point = False
                    if max_index != False:
                        if np.abs((h + k)) > max_index:
                            omit_point = True
                    for each in omit:
                        if h == each[0]:
                            if k == each[1]:
                                if l == each[2]:
                                    omit_point = True
                                    
                    if omit_point == False:       
           
                        hkl = (h,k,l)
                        projx,projy, line= equatorial_projection(plane_vector(hkl, latt_a,latt_b,latt_c))
                        ax2.scatter(projx,projy, marker = 'o', color = 'black')
                        projx,projy = np.round(projx, 2), np.round(projy, 2)

                        if show_trace == True:
                            xu,yu= equatorial_trace_projection(plane_sphere_intercept(hkl, latt_a, latt_b, latt_c))
                            ax2.plot(xu,yu, color='black')

                        pole_label = str(hkl)
                        if four_index_notation == True:
                            #Whether the intellectual cost of the four-index system over the usual three-index notation is justified by its elegance is a matter of taste.
                            pole_label = str((h,k,-(h+k),l))
                        try:
                            pos_y = label_dict[projx]
                            try:
                                label_list = pos_y[projy]
                                label_list.append(pole_label)
                            except:
                                pos_y[projy] = [pole_label]
                        except:
                            label_dict[projx] = {projy:[pole_label]}
                        

                    x,y,z = plane_sphere_intercept((0,0,0), latt_a, latt_b, latt_c)
                    ax2.plot(x,y, color='black', lw = 1)
    if show_label == True:                
        for projx in label_dict.keys():
            for projy in label_dict[projx].keys():
                label = ''
                for hkl in label_dict[projx][projy]:
                    label += hkl + ', '
                label.strip(', ')
                ax2.text(projx,projy+0.05,label, color = 'red', fontsize = 6)


    plt.show()
    
def show_hkl(lattice_vectors, hkl,scale):
    latt_a, latt_b, latt_c = lattice_vectors
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[:, :], projection='3d')
    
    x,y,z = lattice(latt_a, latt_b, latt_c, 1)
    ax.plot_surface(x,y,z, color='grey', alpha = 0.3)
    x,y,z,u,v,w = plane_vector(hkl, latt_a,latt_b,latt_c)
    ax.quiver(0,0,0,u,v,w, color = 'red')
    x,y,z = plane(hkl, latt_a, latt_b, latt_c, scale = scale)
    ax.plot_surface(x,y,z, color='blue', alpha = 0.3)
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
def within_bound_region(lattice_vectors, region_bounds, point, show = False):
    ####plot###
    latt_a, latt_b, latt_c = lattice_vectors
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    if show == True:
        fig = plt.figure(constrained_layout=False)
        gs = fig.add_gridspec(nrows=2, ncols=3,wspace=0.7)
        ax = fig.add_subplot(gs[:, :-1], projection='3d')
    
    ####end####
    upper_bounding_points = []
    lower_bounding_points = []
    for hkl in region_bounds:
        x, y, z, u,v,w  = plane_vector(hkl, latt_a,latt_b,latt_c)
        u_normal = np.array((u[0], v[0], np.abs(w[0])))
        l_normal = np.array((u[0], v[0], -np.abs(w[0])))
        upper_bounding_points.append(u_normal)
        lower_bounding_points.append(l_normal)
    within = False
    opposite = False
    for bounding_points in [upper_bounding_points, lower_bounding_points]:
        b1, b2, b3 = bounding_points[0], bounding_points[1], bounding_points[2]
        b12, b23, b31 = b2-b1, b3-b2, b1- b3
        len_b12, len_b23, len_b31 = np.linalg.norm(b12), np.linalg.norm(b23), np.linalg.norm(b31)
        bound_plane_normal = np.cross(b12, b23)
        if np.dot(bound_plane_normal, b1) < 0:
            bound_plane_normal *= -1
        bound_plane_normal = 1/np.linalg.norm(bound_plane_normal)*bound_plane_normal
        d = -np.dot(bound_plane_normal, b31)
        if (np.dot(point, bound_plane_normal)) == 0:
            factor = 0
        else:
            factor = np.dot(b1, bound_plane_normal)/(np.dot(point, bound_plane_normal)) #the intercept of the position vector of the point with the bound plane
        p = factor * point 
        if factor <= 0:
            opposite = True
        elif factor > 0:
            if np.sign(bound_plane_normal[2]) == np.sign(point[2]):
                opposite = False
            else:
                opposite = True
            
            
        ##plot for checking
        if show == True:
            ax.plot([b1[0],b2[0],b3[0], b1[0]], [b1[1],b2[1],b3[1], b1[1]], [b1[2],b2[2],b3[2], b1[2]], color = 'blue')
            ax.plot([0,bound_plane_normal[0]], [0,bound_plane_normal[1]], [0,bound_plane_normal[2]], color = 'blue', ls = '--')
            xsp,ysp,zsp = sphere(1,50)
            ax.plot_surface(xsp,ysp,zsp, color='grey', alpha = 0.05)
            ax.plot([point[0]], [point[1]], [point[2]], color= 'red', marker= 'x')
            ax.plot([0,point[0]], [0,point[1]], [0,point[2]], color= 'red')
            if opposite == False:
                ax.plot([0,point[0],p[0]], [0,point[1],p[1]], [0,point[2],p[2]], color= 'red')
                ax.plot([p[0]], [p[1]], [p[2]], color= 'red', marker= 'o')
            show_equatorial_region(region_bounds, latt_a,latt_b,latt_c, ax)
            lprojx,lprojy, linel= equatorial_projection([0,0,0,[point[0]],[point[1]],[point[2]]])
            ax.scatter(lprojx, lprojy, color = 'grey', marker = 'o') 
            head,tail = linel
            ax.plot((head[0], tail[0]),(head[1], tail[1]),(head[2], tail[2]), ls = '--', color = 'grey')

        #####end of plot
        
        rot_axis = np.array((0.5*bound_plane_normal[0], 0.5*bound_plane_normal[1], 0.5*bound_plane_normal[2]+0.5))
        if np.linalg.norm(rot_axis) == 0:
            rot_axis = np.array((0,0,1))
        normalise = 1/np.linalg.norm(rot_axis)
        rot_axis = normalise*rot_axis
        b1 = rotate_180(rot_axis, b1)
        b2 = rotate_180(rot_axis, b2)
        b3 = rotate_180(rot_axis, b3)
        p = rotate_180(rot_axis, p)
        b12, b23, b31 = b2-b1, b3-b2, b1- b3

        i12p = (b12[0]*p[1]- b12[1]*p[0])/(b1[1]*b12[0]-b1[0]*b12[1])- 1
        i123 = (b12[0]*b3[1]- b12[1]*b3[0])/(b1[1]*b12[0]-b1[0]*b12[1])-1
        i23p = (b23[0]*p[1]- b23[1]*p[0])/(b2[1]*b23[0]-b2[0]*b23[1]) - 1
        i231 = (b23[0]*b1[1]- b23[1]*b1[0])/(b2[1]*b23[0]-b2[0]*b23[1]) - 1
        i31p = (b31[0]*p[1]- b31[1]*p[0])/(b3[1]*b31[0]-b3[0]*b31[1]) - 1
        i312 = (b31[0]*b2[1]- b31[1]*b2[0])/(b3[1]*b31[0]-b3[0]*b31[1]) - 1

        if i12p*i123 >= 0:
            if i23p*i231 >= 0:
                if i31p*i312 >= 0:
                    if opposite == False:
                        within = True
    return within

def show_equatorial_region(region_bounds, latt_a,latt_b,latt_c, ax):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    sphere_intercepts = []
    for hkl in region_bounds:
        plane = plane_vector(hkl, latt_a,latt_b,latt_c)
        projx,projy, line= equatorial_projection(plane)
        ax.scatter(projx,projy, marker = 'o', color = 'black')
        x,y,z,u,v,w = plane
        sphere_intercepts.append(np.asarray((u[0],v[0],w[0])))
    for i in range(len(sphere_intercepts)):
        p1 = np.asarray(sphere_intercepts[i])
        for j in range(len(sphere_intercepts)):
            p2 = np.asarray(sphere_intercepts[j])
            xx,yy,zz = shortest_path(p1,p2, hemisphere_correct = False)
            xu, yu = equatorial_trace_projection([xx,yy,zz])
            ax.plot(xu,yu, color='grey')

def stereograph_sample_region(lattice_vectors, region_bounds, ranges = range(-3,3), threshold_hkl = 10, threshold_d = 0, factor = 1e-6):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=2,wspace=0.2)
    ax2 = fig.add_subplot(gs[:, :])


    latt_a, latt_b, latt_c = lattice_vectors
    latt_a, latt_b, latt_c =latt_a*factor, latt_b*factor, latt_c*factor
    
    x,y,z = plane_sphere_intercept((0,0,0), latt_a, latt_b, latt_c)
    ax2.plot(x,y, color='black', lw = 1)
    
    sphere_intercepts = []
    
    for hkl in region_bounds:
        plane = plane_vector(hkl, latt_a,latt_b,latt_c)
        projx,projy, line= equatorial_projection(plane)
        ax2.scatter(projx,projy, marker = 'o', color = 'black')
        projx,projy = np.round(projx, 2), np.round(projy, 2)
        x,y,z,u,v,w = plane
        sphere_intercepts.append(np.asarray((u[0],v[0],w[0])))
    for i in range(len(sphere_intercepts)):
        p1 = np.asarray(sphere_intercepts[i])
        for j in range(len(sphere_intercepts)):
            p2 = np.asarray(sphere_intercepts[j])
            xx,yy,zz = shortest_path(p1,p2, hemisphere_correct = False)
            xu, yu = equatorial_trace_projection([xx,yy,zz])
            ax2.plot(xu,yu, color='red')
    count =0 
    for h in ranges:
        for k in ranges:
            for l in ranges:
                if np.sqrt(h**2 + k**2 + l**2) < threshold_hkl:
                    hkl = [h,k,l]
                    x,y,z,u,v,w  = plane_vector(hkl, latt_a,latt_b,latt_c)
                    vector = np.array((u[0],v[0],w[0]))
                    within = within_bound_region(lattice_vectors, region_bounds, vector)
                    if within == True:
                        vector = np.array((0,0,0,u,v,w))
                        projx,projy, line= equatorial_projection(vector)
                        ax2.scatter(projx,projy, marker = 'o', color = 'black')
                        print('Within Region: ', h,k,l,', Threshold Factor: ', np.sqrt(h**2 + k**2 + l**2))
                        count += 1
    print('Total: ', count)


    plt.show()
    
def rotate_180(rot_axis, point):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    u,v,w = rot_axis[0], rot_axis[1], rot_axis[2]
    x,y,z = point[0], point[1], point[2]
    nx, ny, nz = (2*u*(u*x + v*y + w*z)-x), (2*v*(u*x + v*y + w*z)- y),(2*w*(u*x + v*y + w*z)- z)
    return np.array((nx, ny, nz))

def test_plane(hkl, lattice_vectors, factor = 1e-6):
    import numpy as np
    latt_a, latt_b, latt_c = lattice_vectors
    latt_a, latt_b, latt_c =latt_a*factor, latt_b*factor, latt_c*factor
    x,y,z,u,v,w  = plane_vector(hkl, latt_a,latt_b,latt_c)
    vector = np.array((u[0],v[0],w[0]))
    return vector
    