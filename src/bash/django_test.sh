LOGIN_URL=http://ai.gemfield.org:5653/admin/login/?next=/
YOUR_USER='admin'
YOUR_PASS='2018'
COOKIES=cookies.txt
CURL_BIN="curl -s -c $COOKIES -b $COOKIES -e $LOGIN_URL"

echo "Django Auth: get csrftoken ..."
$CURL_BIN $LOGIN_URL > /dev/null
DJANGO_TOKEN="csrfmiddlewaretoken=$(grep csrftoken $COOKIES | sed 's/^.*csrftoken\s*//')"
echo "got csrftoken: $DJANGO_TOKEN"
echo "Perform login ..."
$CURL_BIN \
    -d "$DJANGO_TOKEN&username=$YOUR_USER&password=$YOUR_PASS" \
    -X POST $LOGIN_URL

echo "Do something while logged in ..."
$CURL_BIN -H \
    -d "$DJANGO_TOKEN" \
    -X POST --data '{"taskopr":"get","taskstatus":"all","taskpage":1,"taskquery":"all_"}' http://ai.gemfield.org:5653/task/

echo -e "\n"
#echo "Logout"
rm $COOKIES
