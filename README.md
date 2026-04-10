# MedTriage Agent (Healthcare RL Environment)

MedTriage Agent is a reinforcement learning environment that simulates a hospital emergency room where an intelligent agent performs the role of a triage nurse. The system is designed to model real-world constraints such as limited ICU beds, limited medical staff, and unpredictable patient inflow. The agent receives patient information including vital signs, symptoms, and medical history, and must make decisions that directly impact patient outcomes.

The primary objective of the agent is to correctly prioritize patients based on severity. This involves identifying critical cases quickly and ensuring that high-risk patients receive immediate attention. At the same time, the agent must efficiently allocate scarce resources such as ICU beds and available doctors without causing preventable patient deterioration or death.

The environment introduces dynamic challenges such as sudden mass-casualty events, where multiple patients arrive at once. This tests the agent’s ability to adapt under pressure and make optimal decisions in highly constrained and time-sensitive situations. The agent must balance fairness, urgency, and resource limitations while maintaining overall system stability.

This project demonstrates the application of reinforcement learning in a high-impact domain. It combines decision-making under uncertainty, multi-constraint optimization, and real-time prioritization. The simulation provides a realistic and scalable framework for experimenting with intelligent healthcare systems.

Overall, MedTriage Agent highlights how AI can be used to support critical decision-making in emergency healthcare settings, making it a strong example of applied reinforcement learning in a socially meaningful context.