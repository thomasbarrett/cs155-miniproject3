all: hmm_trainer

hmm_trainer: hmm_trainer.c
	$(CC) hmm_trainer.c -o hmm_trainer -O3

clean:
	rm hmm_trainer