!pip install pytelegrambotapi
import telebot
import cv2

#TOKEN = "1125736291:AAFkW8H2jbm7Egc22uYkhHBxR6Lru2SSF-Q"
bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
	bot.reply_to(message, "Welcome, send me the leaf image to be tested:")

@bot.message_handler(content_types=['photo'])
def photo(message):
    bot.reply_to(message,"The disease results are")
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    main_img = cv2.imread('image.jpg')
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        
        #Color features
    red_channel = img[:,:,0]
    green_channel = img[:,:,1]
    blue_channel = img[:,:,2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0
    red_mean = np.mean(red_channel)
    #print(red_mean)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)
        
        #Texture features
    textures = mt.features.haralick(gs)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]

                   
        
    vector =[[red_mean,green_mean,blue_mean,red_std,green_std,blue_std, contrast,correlation,inverse_diff_moments,entropy]  ]
    vector=pd.DataFrame(vector)

    #y_preda=np.dot(wa,vector)+ba
    print(vector)
    print(vector.shape)
    
    vector = scaler.transform(vector)
    sk=model.predict(np.asarray(vector))
    v2=vector.T.ravel()
    print(v2)
    y_predg=1.0/(1.0+np.exp(-1*(np.dot(wg,v2)+bg)))

    y_preda=1.0/(1.0+np.exp(-1*(np.dot(wa,v2)+ba)))
    # print(y_pred)
    #print(vector)
    print(sk.shape)
    print(sk)
    #y_predq = sn.sigmoid(sn.perceptron(vector.reshape(1,0)))
    #res=[str(sk>0.5),str(y_predg>0.5),str(y_preda>0.5)]
    
    aa=str(sk>0.5)
    bb=str(y_predg>0.5)
    cc=str(y_preda>0.5)
    bot.send_message(chat_id=message.chat.id, text= "using SKIT Learn : "+aa[1:len(aa)-1])
    bot.send_message(chat_id=message.chat.id, text= "using GD         : "+bb[1:len(bb)-1])
    bot.send_message(chat_id=message.chat.id, text= "using Adam       : "+cc[1:len(cc)-1])
    
    #result=np.dot(vector, wa) + ba
    #y_pred=1.0/(1.0 + np.exp(-1*result))
    # Returns a NumPy Array
    # Predict for One Observation (image)
    
    

bot.polling()
