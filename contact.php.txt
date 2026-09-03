<?php
header('Content-Type: application/json');
if ($_SERVER['REQUEST_METHOD'] !== 'POST') { http_response_code(405); echo json_encode(['ok'=>false,'msg'=>'Method not allowed']); exit; }
if (!empty($_POST['website'])) { echo json_encode(['ok'=>true]); exit; } // bot honeypot

$name    = trim($_POST['name'] ?? '');
$email   = trim($_POST['email'] ?? '');
$org     = trim($_POST['organization'] ?? '');
$role    = trim($_POST['role'] ?? '');
$message = trim($_POST['message'] ?? '');

if ($name === '' || $message === '' || !filter_var($email, FILTER_VALIDATE_EMAIL)) {
    http_response_code(400);
    echo json_encode(['ok'=>false,'msg'=>'Please provide a valid name, email and message.']); exit;
}
$name  = str_replace(["\r","\n"], ' ', substr($name,0,120));
$email = str_replace(["\r","\n"], ' ', substr($email,0,120));

$to      = 'ghiyasifarreza@gmail.com';
$subject = 'RezpharmaCDSS.me inquiry — ' . $name;
$body    = "Name: $name\nEmail: $email\nOrganization: $org\nRole: $role\n\nMessage:\n$message\n\n--\nSent from rezpharmacdss.me";
$headers = "From: rezpharmacdss.me <noreply@rezpharmacdss.me>\r\nReply-To: $email\r\nContent-Type: text/plain; charset=UTF-8\r\n";

if (mail($to, $subject, $body, $headers)) { echo json_encode(['ok'=>true]); }
else { http_response_code(500); echo json_encode(['ok'=>false,'msg'=>'Mail server error — please email ghiyasifarreza@gmail.com directly.']); }