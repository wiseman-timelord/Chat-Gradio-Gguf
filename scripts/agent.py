# scripts/agent.py

import time
import threading
import json
from scripts import temporary
from scripts.utility import speak_text
from scripts.models import unload_models, get_agentic_response
def run_agent_loop(goal: str, llm_state, models_loaded_state, update_ui_callback):
    """
    Main loop for the agent's operation.
    This function will be executed in a separate thread.

    Args:
        goal (str): The user-defined goal.
        llm_state: The loaded language model instance.
        models_loaded_state (bool): Flag indicating if a model is loaded.
        update_ui_callback (function): A function to send updates back to the Gradio UI.
    """
    if not models_loaded_state:
        update_ui_callback(tasks="Error: Model not loaded.", log="Please load a model on the Configuration page before starting the agent.")
        return

    update_ui_callback(log="Agent starting...")
    temporary.cancel_agent_loop = False
    temporary.pause_agent_loop = False

    history = []

    # 1. Decompose goal into tasks
    update_ui_callback(log="Decomposing goal into tasks...")
    task_prompt = f"Based on the goal '{goal}', generate a concise, numbered list of tasks to accomplish it. Do not explain the tasks. For example: '1. Task one.\n2. Task two.'"
    history.append({"role": "user", "content": task_prompt})

    tasks_response = ""
    for chunk in get_agentic_response(history, llm_state, models_loaded_state, is_agent=True):
        tasks_response += chunk

    tasks = [t.strip() for t in tasks_response.split('\n') if t.strip()]
    history.append({"role": "assistant", "content": tasks_response})
    update_ui_callback(tasks='\n'.join(tasks), log="Task list created.")

    completed_tasks = []
    iteration_count = 0
    unload_timer_start = None
    model_unloaded = False

    # 2. Execute task loop
    while tasks:
        while temporary.pause_agent_loop:
            if unload_timer_start is None:
                unload_timer_start = time.time()
                update_ui_callback(log="Agent paused. Model will unload in 60s.")

            if time.time() - unload_timer_start > 60 and not model_unloaded and models_loaded_state:
                update_ui_callback(log="Unloading model due to inactivity...")
                status, llm_state, models_loaded_state = unload_models(llm_state, models_loaded_state)
                model_unloaded = True
                update_ui_callback(log=status)

            time.sleep(0.5)

        if unload_timer_start is not None:
            unload_timer_start = None
            model_unloaded = False
            update_ui_callback(log="Agent resumed.")

        if temporary.cancel_agent_loop:
            update_ui_callback(log="Agent loop cancelled by user.")
            break

        current_task = tasks.pop(0)
        iteration_count += 1

        task_display = '\n'.join([f"[✓] {t}" for t in completed_tasks] + [f"[►] {current_task}"] + tasks)
        update_ui_callback(tasks=task_display, log=f"Executing task: {current_task}")

        if "user input" in current_task.lower():
            update_ui_callback(log="Waiting for user input... (120s timeout)")
            timer_start = time.time()
            while time.time() - timer_start < 120:
                if not temporary.pause_agent_loop: # User has not paused manually
                    time.sleep(0.5)
                else:
                    break # User paused, break inner loop
            if not temporary.pause_agent_loop: # Timer expired
                temporary.pause_agent_loop = True
                update_ui_callback(log="User input timed out. Agent paused.", agent_state="paused")
        else:
            execute_task(current_task, history, llm_state, models_loaded_state, update_ui_callback)

        completed_tasks.append(current_task)

        if iteration_count % 5 == 0 and iteration_count > 0:
            summary_message = f"Progress update: {len(completed_tasks)} of {len(completed_tasks) + len(tasks)} tasks complete."
            update_ui_callback(log=summary_message)
            try:
                speak_text(summary_message)
            except Exception as e:
                update_ui_callback(log=f"Speech Error: {e}")

    if not temporary.cancel_agent_loop:
        task_display = '\n'.join([f"[✓] {t}" for t in completed_tasks])
        update_ui_callback(tasks=task_display, log="All tasks completed. Goal achieved.")

    temporary.cancel_agent_loop = False
    temporary.pause_agent_loop = False

def execute_task(task, history, llm_state, models_loaded_state, update_ui_callback):
    """Executes a single task by calling the LLM."""
    history.append({"role": "user", "content": f"Execute this task: {task}"})

    response = ""
    first_chunk = True
    for chunk in get_agentic_response(history, llm_state, models_loaded_state, is_agent=True):
        response += chunk
        if first_chunk:
            update_ui_callback(log=chunk, log_type="full")
            first_chunk = False
        else:
            update_ui_callback(log=chunk, log_type="chunk")

    history.append({"role": "assistant", "content": response})
