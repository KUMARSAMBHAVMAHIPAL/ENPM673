import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse as agp

def parse_arg():
    parser = agp.ArgumentParser()
    parser.add_argument('--datapath',default = './data/data_1.csv', help = 'give the relative path of the dataset' )
    return parser.parse_args()


def fit3points(points):
    
    assert len(points)==3
    A = np.array(
    [
        [points[0][0]**2,points[0][0],1],
        [points[1][0]**2,points[1][0],1],
        [points[2][0]**2,points[2][0],1],
        
    ])

    B = np.array(
        [
            [points[0][1]],
            [points[1][1]],
            [points[2][1]],
        ]
    )

    try:
        X = np.matmul(np.linalg.inv(A),B).flatten()
    except:
        X= np.array([0,0,0])
    return X

def inregion(point, curve,offset=10):
    
    assert len(curve)==3
    assert len(point)==2
    
    if abs(curve[0]*point[0]**2+curve[1]*point[0]+curve[2]-point[1])<=offset:
        return True
    return False

def ransac(points,offset=10, rounds= 100):
    
    try:
        assert type(point) == np.array
    except:
        points= np.array(points)
    
    curve = [0,0,0]
    prevper = 0
    for _ in range(rounds):
        
        point3 = np.zeros((3,2))
        point3[0] = points[np.random.randint(1,len(points))]
        point3[1] = points[np.random.randint(1,len(points))]
        point3[2] = points[np.random.randint(1,len(points))]

        point3 = np.array(point3)
        
#         print(point3)
        assert point3.shape==(3,2)
        cur_curve = fit3points(point3)
        count=0
        for i in points:
            if inregion(i,cur_curve, offset=offset):
                count+=1
        curper = count/len(points)
        if curper>prevper:
        	prevper=curper
        	curve=cur_curve
    new_points=[]
    for i in points:
        if inregion(i,curve, offset=offset):
        	new_points.append(list(i))

    print("Percentage of Inliers: ",prevper)
    return curve, np.array(new_points)

def ls_curve(data):
    A = [data[:,0]**2,data[:,0],[1]*len(data)]
    A = np.array(A).T
    B = data[:,1]
    B = np.array(B).T
    return np.linalg.inv(A.T.dot(A)).dot(A.T.dot(B))

def visualize_ransac(data, offset=10, figsize=(15,9), rounds=1000):
    points = [data[:,0],data[:,1]]
    points= np.array(points).T
    
    X,points = ransac(points, offset=offset, rounds=rounds)
    X1 = ls_curve(points)
    
    print("The parameters produced by RANSAC:",X)
    print("The parameters produced by LS after RANSAC:",X1)
    fitted_x =  np.linspace(min(data[:,0]),max(data[:,0]), 100)
    fitted_y = np.array(list(X[0]*i**2+X[1]*i+X[2] for i in fitted_x))
    fitted_y1 = np.array(list(X[0]*i**2+X[1]*i+X[2]+offset for i in fitted_x))
    fitted_y2 = np.array(list(X[0]*i**2+X[1]*i+X[2]-offset for i in fitted_x))
    fitted_yf = np.array(list(X1[0]*i**2+X1[1]*i+X1[2] for i in fitted_x))
    
    plt.figure(figsize=figsize)

    plt.scatter(data[:,0],data[:,1],s=2)
    plt.scatter(sum(data[:,0])/len(data[:,0]),sum(data[:,1])/len(data[:,0]),color='green',label='Mean of Datapoints' )
    plt.scatter(sum(points[:,0])/len(points[:,0]),sum(points[:,1])/len(points[:,0]),s=8,color='yellow',label='Mean of Datapoints after RANSAC' )
    plt.plot(fitted_x,fitted_yf,'k', label='LS Curve after RANSAC')
    plt.plot(fitted_x,fitted_y, 'r:', label= 'RANSAC Curve')
    plt.plot(fitted_x,fitted_y1,':', label='Upper Bound of Offset')
    plt.plot(fitted_x,fitted_y2,':', label="Lower Bound of Offset")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":

    args = parse_arg()

    points = []
    with open(args.datapath, 'r' ,newline='') as f:
        data = csv.reader(f)
        for i in data:
            points.append(i)
    points = points[1:]
    points = np.array(points, dtype=np.float)

    params = {'./data/data_1.csv':{'offset':35,'rounds':1000},'./data/data_2.csv':{'offset':35,'rounds':1000}}

    visualize_ransac(points,**params[args.datapath])