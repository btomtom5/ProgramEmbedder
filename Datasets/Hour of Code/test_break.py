for i in range(100):
	print('looping')
	try:
		raise Exception("raise an Exception")
	except Exception:
		break
