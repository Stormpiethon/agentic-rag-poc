"""
Every time an agent makes a decision or queries pinecone, a detailed, step-by-step log of 
the decision-making process is generated. This log includes information about the agent's 
reasoning, the data it accessed, and the actions it took.
"""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor


def init_telemetry():
    """Initializes OpenTelemetry tracing for the LangChain/LangGraph application."""
    # Setup the basic Tracer Provider
    provider = TracerProvider()
    
    # Configure the processor to print traces cleanly to your terminal console
    # In a production app, this would send data to Datadog, AWS X-Ray, or Honeycomb.
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    
    # Set the global tracer provider
    trace.set_tracer_provider(provider)
    
    # This automatically hooks into all chains, agents, and tool calls.
    LangchainInstrumentor().instrument()
    
    print("Telemetry and tracing hooks successfully initialized.")