for lr in 0.005
do
        run_name="binary_070120_$lr"
        epochs=2
        python main_binary.py "${run_name}" "$lr" "${epochs}" 
done
