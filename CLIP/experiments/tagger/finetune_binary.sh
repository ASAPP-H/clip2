for lr in 0.005
do
        run_name="binary$lr"
        epochs=30
        python main_binary.py "${run_name}" "$lr" "${epochs}" 
done
