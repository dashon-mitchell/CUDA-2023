// gcc N-BodiesInBox.c -o N-BodiesInBox -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NUMSPHERES 7 // this has to be atleast 2
#define XWindowSize 1000
#define YWindowSize 1000

#define STOP_TIME 10000.0
#define DT        0.0001

#define GRAVITY 0.1 

#define MASS 10.0  	
#define DIAMETER 1.0

#define SPRING_STRENGTH 50.0
#define SPRING_REDUCTION 0.01

#define DAMP 0.0

#define DRAW 10

#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 0.5

const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

// Globals
struct SphereStruct // Structure declaration
{   
	float px,py,pz;  // sphere position
	float vx,vy,vz; // sphere velocity
	float fx,fy,fz; // sphere forces
	float mass;
};
struct SphereStruct *Sphere;
void set()
{
	for(int i = 0; i < NUMSPHERES; i++)
	{
		Sphere[i].fx=0;
		Sphere[i].fy=0;
		Sphere[i].fz=0;
	}
}

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	int yeahBuddy;
	float dx, dy, dz, seperation;
	Sphere=(struct SphereStruct*)malloc(NUMSPHERES*sizeof(struct SphereStruct));
	
	for(int i = 0; i < NUMSPHERES; i++)
	{
		Sphere[i].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		Sphere[i].py = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		Sphere[i].pz = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		
		Sphere[i].vx = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Sphere[i].vy = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Sphere[i].vz = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Sphere[i].mass=1.0;
		
		yeahBuddy = 0;
		while(yeahBuddy == 0  && i!=0)
		{
			
			dx = Sphere[i].px -Sphere[i-1].px;
			dy = Sphere[i].py -Sphere[i-1].py;
			dz = Sphere[i].pz -Sphere[i-1].pz;
			seperation = sqrt(dx*dx + dy*dy + dz*dz);
			yeahBuddy = 1;
			if(seperation < DIAMETER) yeahBuddy = 0;
		}
		
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	



		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	for(int i = 0; i < NUMSPHERES; i++)
	{
		glColor3d(1.0,0.5,1.0);
		glPushMatrix();
		glTranslatef(Sphere[i].px,Sphere[i].py,Sphere[i].pz);
		glutSolidSphere(radius,20,20);
		glPopMatrix();
	}
		
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	for(int i = 0; i < NUMSPHERES; i++)
	{
		if(Sphere[i].px > halfBoxLength)
		{
			Sphere[i].px = 2.0*halfBoxLength - Sphere[i].px;
			Sphere[i].vx = - Sphere[i].vx;
		}
		else if(Sphere[i].px < -halfBoxLength)
		{
			Sphere[i].px = -2.0*halfBoxLength - Sphere[i].px;
			Sphere[i].vx = - Sphere[i].vx;
		}
		
		if(Sphere[i].py > halfBoxLength)
		{
			Sphere[i].py = 2.0*halfBoxLength - Sphere[i].py;
			Sphere[i].vy = - Sphere[i].vy;
		}
		else if(Sphere[i].py < -halfBoxLength)
		{
			Sphere[i].py = -2.0*halfBoxLength - Sphere[i].py;
			Sphere[i].vy = - Sphere[i].vy;
		}
				
		if(Sphere[i].pz > halfBoxLength)
		{
			Sphere[i].pz = 2.0*halfBoxLength - Sphere[i].pz;
			Sphere[i].vz = - Sphere[i].vz;
		}
		else if(Sphere[i].pz < -halfBoxLength)
		{
			Sphere[i].pz = -2.0*halfBoxLength - Sphere[i].pz;
			Sphere[i].vz = - Sphere[i].vz;
		}
	}
}

void get_forces()
{
	int j0=1;
	int j=1;
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
	for(int i = 0; i < NUMSPHERES; i++)
	{
		int j=j0;
		while(j<=NUMSPHERES)
		{
			if(i!=j)
			{
				dx = Sphere[j].px - Sphere[i].px;
				dy = Sphere[j].py - Sphere[i].py;
				dz = Sphere[j].pz - Sphere[i].pz;
							
				r2 = dx*dx + dy*dy + dz*dz;
				r = sqrt(r2);

				forceMag =  Sphere[j].mass*Sphere[i].mass*GRAVITY/r2;
						
				if (r < DIAMETER)
				{
					dvx = Sphere[j].vx - Sphere[i].vx;
					dvy = Sphere[j].vy - Sphere[i].vy;
					dvz = Sphere[j].vz - Sphere[i].vz;
					inout = dx*dvx + dy*dvy + dz*dvz;
					if(inout <= 0.0)
					{
						forceMag +=  SPRING_STRENGTH*(r - DIAMETER);
					}
					else
					{
						forceMag +=  SPRING_REDUCTION*SPRING_STRENGTH*(r - DIAMETER);
					}
				}

				Sphere[i].fx += forceMag*dx/r;
				Sphere[i].fy += forceMag*dy/r;
				Sphere[i].fz += forceMag*dz/r;
				Sphere[j].fx += -forceMag*dx/r;
				Sphere[j].fy += -forceMag*dy/r;
				Sphere[j].fz += -forceMag*dz/r;
			}
			j++;
		}
		j0++;
	}
}

void move_bodies(float time)
{
	for(int i = 0; i < NUMSPHERES; i++)
	{
		if(time == 0.0)
		{
			Sphere[i].vx += 0.5*DT*(Sphere[i].fx - DAMP*Sphere[i].vx)/Sphere[i].mass;
			Sphere[i].vy += 0.5*DT*(Sphere[i].fy - DAMP*Sphere[i].vy)/Sphere[i].mass;
			Sphere[i].vz += 0.5*DT*(Sphere[i].fz - DAMP*Sphere[i].vz)/Sphere[i].mass;
			
		}
		else
		{
			Sphere[i].vx += DT*(Sphere[i].fx - DAMP*Sphere[i].vx)/Sphere[i].mass;
			Sphere[i].vy += DT*(Sphere[i].fy - DAMP*Sphere[i].vy)/Sphere[i].mass;
			Sphere[i].vz += DT*(Sphere[i].fz - DAMP*Sphere[i].vz)/Sphere[i].mass;
			
		}

		Sphere[i].px += DT*Sphere[i].vx;
		Sphere[i].py += DT*Sphere[i].vy;
		Sphere[i].pz += DT*Sphere[i].vz;

		
		keep_in_box();
	}
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	set();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}
void clean_up()
{
	free(Sphere);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Rumbling, Rumbling its coming Rumbling, RUmbling, Beware");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	clean_up();
	return 0;
}
