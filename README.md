![Recording 2024-01-27 212416](https://github.com/Rakshya8/NLP_Assignments/assets/45217500/39f02a3f-eb63-411f-a16f-6bd5b032f160)# Assignment 2: Language Model

## Table of Contents

1. [Introduction](#1-introduction)
    - [Purpose of the Web Application](#11-purpose-of-the-web-application)
    - [Overview of Features](#12-overview-of-features)

2. [Architecture Overview](#2-architecture-overview)
    - [Frontend Components](#21-frontend-components)
    - [Backend Components](#22-backend-components)
    - [Language Model Integration](#23-language-model-integration)

3. [Components Description](#3-components-description)
    - [Frontend Components](#31-frontend-components)
        - [HTML/CSS Templates](#311-htmlcss-templates)
        - [Flask Views](#312-flask-views)
    - [Backend Components](#32-backend-components)
        - [Flask Application](#321-flask-application)
        - [Language Model Module](#322-language-model-module)

4. [Data Flow](#4-data-flow)
    - [User Input Processing](#41-user-input-processing)
    - [Communication with the Language Model](#42-communication-with-the-language-model)
    - [Displaying Results](#43-displaying-results)

5. [Integration with Language Model](#5-integration-with-language-model)
    - [Loading the Language Model](#51-loading-the-language-model)
    - [Generating Text](#52-generating-text)
    - [Handling Temperature Settings](#53-handling-temperature-settings)
    - [Web Interface and Language Model Interaction](#54-web-interface-and-language-model-interaction)

6. [User Interaction](#6-user-interaction)
    - [A1: Similar Words](#61-a1-similar-words)
    - [A2: Language Model Text Generation](#62-a2-language-model-text-generation)


## 1. Introduction

### 1.1 Purpose of the Web Application

The web application serves as an interface for users to interact with a language model. It provides functionalities such as finding similar words and generating text based on user prompts.

### 1.2 Overview of Features

- **A1: Similar Words:** Allows users to find words similar to a given input using pre-trained word embeddings.
- **A2: Language Model Text Generation:** Enables users to generate text based on a prompt using a pre-trained LSTM language model.

## 2. Architecture Overview

### 2.1 Frontend Components

- **HTML/CSS Templates:** Provide the structure and style for the web pages.
- **Flask Views:** Define the routes and handle user requests.

### 2.2 Backend Components

- **Flask Application:** Manages the backend logic and communication between frontend and backend.
- **Language Model Module:** Handles interactions with the pre-trained language model.

### 2.3 Language Model Integration

The web application integrates with a pre-trained language model for text generation. The model is loaded and used to generate text based on user prompts.

## 3. Components Description

### 3.1 Frontend Components

#### 3.1.1 HTML/CSS Templates

- `index.html`: Main landing page.
- `a1.html`: Page for A1 functionality.
- `a2.html`: Page for A2 functionality.

#### 3.1.2 Flask Views

- `home()`: Renders the main landing page.
- `a1()`: Handles requests and renders A1 functionality.
- `a2()`: Handles requests and renders A2 functionality.

### 3.2 Backend Components

#### 3.2.1 Flask Application

- `app.py`: Main Flask application file.
- `templates/`: Directory containing HTML templates.
- `static/`: Directory containing CSS and other static files.

#### 3.2.2 Language Model Module

- `lstm.py`: Defines the LSTM language model class and related functions.

## 4. Data Flow

### 4.1 User Input Processing

1. User submits a form on the web page.
2. Flask views process the form data.
3. User input is sent to the backend for further processing.

### 4.2 Communication with the Language Model

1. Language model is loaded during the application startup.
2. User input is passed to the language model for text generation.
3. Generated text is returned to the backend.

### 4.3 Displaying Results

1. Backend sends the generated results to the frontend.
2. Frontend dynamically updates the web page with the results.

## 5. Integration with Language Model

### 5.1 Loading the Language Model

- The LSTM language model is loaded from a pre-trained checkpoint during the application startup.

### 5.2 Generating Text

- The language model is used to generate text based on user prompts, incorporating temperature settings for diversity.

### 5.3 Handling Temperature Settings

- Temperature settings are passed to the language model during text generation to control the randomness of the output.

### 5.4 Web Interface and Language Model Interaction

The web application interfaces with the language model in the following ways:

- **A2 Functionality (Language Model Text Generation):**
  - The user inputs a prompt.
  - The backend passes the prompt to the pre-trained LSTM language model.
  - Generated text is returned to the frontend for display.

## 6. User Interaction

### 6.1 A1: Similar Words

- Users input a word in A1 and receive a list of words similar to the input, based on pre-trained embeddings.

### 6.2 A2: Language Model Text Generation

- Users input a prompt in A2 and receive generated text from the language model.


## Demo of website

![Recording 2024-01-27 212416](https://github.com/Rakshya8/NLP_Assignments/assets/45217500/4abcd6ee-541f-42f3-a85e-dfa3a3b0153a)


This detailed documentation outlines how the web application interfaces with the language model, covering user interaction, data flow, and integration specifics.
