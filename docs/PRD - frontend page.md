# PRD - frontend page

## Overview

This PRD aims to define the spec of the page where a user can classify their "Nui" for which character.

## Product Overview

This product consists of 2 pages

1. Home page
2. Nui camera page

### Home Page

A user will land on this page on the initial render. On this page, a user can

* Upload an image and see the result of the classification.
  * This is helpful for the initial development for easy verification on a PC without having a camera functionality.
* A user can open a camera page to take the Nui in real time for the classification.
  * This is the main feature of the product

### Camera Page

A user can open a camera view and frame a Nui (or Nui's) to classify them. The app automatically classify Nui and show the result.

MVP: We can just show the result as a simple string (Saki, Kotone, etc) on the top of the screen.

Nice to have: We cans how a frame (frames) of the position where the Nui is placed (with a green square). And we can classify 1+ Nui's.

Nice to have: The color of the square will be changed by the type of Nui that we already defined (Well-know character color like Red for Saki, Yellow for Kotone, etc.).

### Classification

A user can execute the classification without a network connection, so the machine learning model should run on user's mobile machine.

### (non-MVP) Feedback system

We want to ask user about the quality of the classification and collect the image and the feedback for our ML ops pipeline to continuously improve our ML model.

## Technical Design

User interfaces should be fully developed with React / Next.js / TypeScript. The styling would be Tailwind.

Use the ONNX Runtime Web for the local ML model. Because the transformation from Pytorch and the ONNX has a native support.

If we need any server side resources, AWS is the only choice.

We assume using a GitHub Pages for hosting.
