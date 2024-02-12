# Painting with neural networks

Required libraries are in requirements.txt file and listed below:

- tensorflow
- keras
- notebook
- numpy
- matplotlib
- pydot

## Backend
For the backend of the web page application that visualizes generated images, 
a simple [Flask](https://flask.palletsprojects.com/en/3.0.x/) application is used.

Only default route __"/"__ with __GET__ method is available. This route parses one argument
called __"input"__, which represents the text prompt to the model. 
Response is an array of 5 images encoded in __base64__ format.

For running the server, using __docker compose__ or building a __docker container__ is 
highly recommended. More about that in the [Docker](#docker) part. 

Manual set-up requires these steps:
1. Have python 3.10 installed
2. Install requirements listed in [requirements.txt](requirements.txt)
3. Make sure you are running the command from the root of the repository, 
as this should be the programs working directory
4. Run
     ```python3 -m flask --app backend/app run -p 3000```
   
   option __-p__ changes the port, as default port _5000_ is often used, which would result in an error
5. Default __GET__ request will respond with __Bad Request__, because of the lack of __"input"__ argument
 
   Try this link: [http://127.0.0.1:3000/?input=red](http://127.0.0.1:8001/?input=red) if that happens
 
## Frontend

Frontend of the application was created with reactive framework called 
[Vue.js](https://vuejs.org/) to enable requesting and re-rendering of generated images
without refreshing the whole page.

It is a simple Single page app, that has a text prompt and a "generate" button. Images
are displayed after the first request to [backend](#backend).

Running the frontend is the easiest using __docker compose__ or building a 
__docker container__.  More about that in the [Docker](#docker) part. 


Manual set-up requires these steps:
1. Have [Node.js](https://nodejs.org/en) installed
2. in the [frontend/](./frontend) folder execute ```npm install```
3. Run ```npm run dev``` after the installation is complete
4. The app should be running right here: http://localhost:5173/

## Docker

To streamline the deployment of the different parts of this project and to avoid 
__architecture__ or __OS__ incompatibilities, we use [Docker](https://www.docker.com/) 
and it's tools. Please make sure you [Docker Engine](https://docs.docker.com/engine/install/)
installed.

The project is set up in two ways, that are somewhat building on each other.

### Docker containers

There are two __Dockerfiles__ present [./Dockerfile_backend](Dockerfile_backend),
[./Dockerfile_frontend](Dockerfile_frontend) For both front and backend respectively.
These provide a set of instructions for building their __Docker Images__.

To build the desired __Docker Image__ you can run at the root repository:
    ```docker build . -t image_name  -f ./Dockerfile_name```.
Option __-f__ is used because the __Dockerfiles__ are named differently then just _"Dockerfile"_
which the command would automatically scan for.

Replacing the placeholers as an example ```docker build . -t frontend-docker  -f ./Dockerfile_frontend```

After the image is build, it can be run via:

* ```Docker run -p 8080:8080 frontend_image_name``` for frontend.
* ```Docker run -p 8001:5000 backend_image_name``` for backend.

The mapping of ports should be kept as is, because that is where the apps are exposed to your machine.

### Docker Compose

With [Docker Compose](https://docs.docker.com/compose/) we can combine the building and running of 
multiple __images__ with a single command and [compose.yaml](compose.yaml) file,
further automating the deployment process. 

You will need to have [Docker Compose](https://docs.docker.com/compose/install/) 
plugin installed in addition to the __Docker Engine__.

When you have __Compose__ installed, simply run ```docker compose up ``` at the same directory 
as the _yaml_ file. After everything is finished, both dockerized services should be up and running. 
You can check in the ["Painting with Neural Nets"](http://127.0.0.1:8080 ) frontend app, 
if it is showing and requests are correctly resolving.



















