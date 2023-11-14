def sendEmail(subject, message, receiver, filenames = []):

    from tools import getPath
    from os.path import basename
    import smtplib, ssl
    from email.mime.base import MIMEBase
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import json

    with open(getPath()["code"] + "email.json", 'r') as f:
        data = json.load(f)

        password = data["password"]
        username = data["username"]
        smtp_server = data["smtp_server"]
        sender_email = data["sender_email"]
        port = data["port"]

    mail = MIMEMultipart()
    mail["From"] = sender_email
    mail["To"] = receiver
    mail["Subject"] = subject

    mail.attach(MIMEText(message, "plain"))

    # attache Results
    for f in filenames:
        fname = getPath()[f[0]]+f[1]
        with open(fname,"rb") as file:
            part = MIMEApplication(file.read(),Name=basename(fname))
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(fname)
        mail.attach(part)

    context = ssl.create_default_context()
    context = ssl._create_unverified_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(username, password)
        server.sendmail(sender_email, receiver, mail.as_string())

    """
    # Example call:
    sendEmail("Simulations has ended", "Simulations has ended", "mail@florian-franke.eu")

    # email.json:
    {
    	"username" : "yourUsername",
    	"password" : "yourPassword",
    	"smtp_server" : "yourServer",
    	"sender_email" : "yourEmail",
    	"port" : 587
    }
    """
