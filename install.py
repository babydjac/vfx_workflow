import launch

if not launch.is_installed("transparent-background"):
    launch.run_pip("install transparent-background==1.3.2", "installing transparent-background for VFX Workflow")
